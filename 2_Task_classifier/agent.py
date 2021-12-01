import os
from google.protobuf.message import EncodeError
import numpy as np
import random
import matplotlib.pyplot as plt
import time

from network import TwinnedQNetwork, DQNPolicy, CateoricalPolicy, LatentNetwork, Classifier
from network.model import Classifier
from utils import disable_gradients, update_params, RunningMeanStats, center_crop, rand_crop, get_aug_image
from memory import Memory

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

class Agent:
    def __init__(
        self,
        env,
        log_dir,
        env_type='custom',
        num_steps=3000000,
        batch_size=12,
        policy_lr=3e-5,
        critic_lr=3e-4,
        latent_lr=1e-4,
        latent_dim=256,
        hidden_units=[1024, 1024, 1024],
        memory_size=30000,
        gamma=0.99,
        multi_step=1,
        target_entropy_ratio=0.98,
        target_update_interval=1,
        use_per=False,
        dueling_net=False,
        log_interval=100,
        num_eval_steps=2000,
        cuda=True,
        seed=0,
        gpu=0,
        num_rollout=2):

        # Data shape
        self.env = env
        self.high_observation_shape = self.env.high_observation_space.shape
        self.low_observation_shape = self.env.low_observation_space.shape
        # self.hybrid_state_shape = self.env.hybrid_state_space.shape
        self.high_action_shape = self.env.high_action_space.shape
        self.num_action = self.high_action_shape[0]
        self.max_step = self.env._max_episode_steps

        # Set seed
        self.seed_all(seed)
        self.action_define()        

        self.latent_dim = latent_dim
        self.leaky_slope = 0.2
        self.beta = 1e-6
        
        # Set device        
        print(f'CUDA available : {torch.cuda.is_available()}')
        self.device = torch.device(
            "cuda:{}".format(gpu) if cuda and torch.cuda.is_available() else "cpu")
        print(self.device)

        # VAE
        self.latent = LatentNetwork(
            self.low_observation_shape[0], self.latent_dim, hidden_units, self.leaky_slope
            ).to(self.device)
        
        # SAC-discrete
        self.policy = CateoricalPolicy(
            self.high_observation_shape[0], self.latent_dim*2, self.num_action
            ).to(self.device)
        self.online_critic = TwinnedQNetwork(
            self.high_observation_shape[0], self.num_action, self.latent_dim*2,
            dueling_net=dueling_net).to(device=self.device)
        self.target_critic = TwinnedQNetwork(
            self.high_observation_shape[0], self.num_action, self.latent_dim*2,
            dueling_net=dueling_net).to(device=self.device).eval()
        
        # DQN
        self.dqn = DQNPolicy(
            self.high_observation_shape[0], self.num_action, self.latent_dim*2,
            dueling_net=dueling_net).to(self.device)
        self.target_dqn = DQNPolicy(
            self.high_observation_shape[0], self.num_action, self.latent_dim*2,
            dueling_net=dueling_net).to(self.device)
        
        # Supervised learning
        self.posb_classifier = Classifier(
            self.high_observation_shape[0], self.latent_dim*2, self.num_action
            ).to(self.device)
        
        self.selc_classifier = Classifier(
            self.high_observation_shape[0], self.latent_dim*2, self.num_action
            ).to(self.device)

        self.target_critic.load_state_dict(self.online_critic.state_dict())
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        disable_gradients(self.target_critic)
        disable_gradients(self.target_dqn)

        self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=critic_lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=critic_lr)
        self.latent_optim = Adam(self.latent.parameters(), lr=latent_lr)
        self.dqn_optim = Adam(self.dqn.parameters(), lr=policy_lr)
        self.posb_classifier_optim = Adam(self.posb_classifier.parameters(), lr=policy_lr)
        self.selc_classifier_optim = Adam(self.selc_classifier.parameters(), lr=policy_lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio)
        self.target_entropy = \
            -np.log(1.0 / self.num_action) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=critic_lr)

        # Offline dataset
        self.memory = Memory(
            memory_size, self.high_observation_shape, self.high_action_shape[0], self.device)
        self.val_memory = Memory(
            memory_size, self.high_observation_shape, self.high_action_shape[0], self.device)
        self.load_data(num_rollout)  


        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)

        # CQL-setting
        self.num_random = 5
        self.temp = 1.0
        self.min_q_weight = 5.0
        self.min_q_version = 2
        self.max_q_backup = False

        self.with_lagrange = False
        if self.with_lagrange:
            self.target_action_gap = 15.0
            self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_prime_optimizer = Adam(
                    [self.log_alpha_prime],
                    lr=3e-4,)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = self.max_step
        self.log_interval = log_interval


        self.latent.load('model/final_latent.pth', self.device)
        
    def action_define(self):
        self.action_dict={"0" : "Laptop close",
                    "1"  : "Drawer open",
                    "2"  : "Drawer close",
                    "3"  : "Box open",
                    "4"  : "Box close",
                    "5"  : "Box push",
                    "6"  : "Green to space",
                    "7"  : "Green to drawer",
                    "8"  : "Orange to space",
                    "9"  : "Orange to box",
                    "10" : "No action left"}
        
        # Check point selection
        self.highest_q = -np.inf
        self.saved_step = []
        self.cur_q_val_history = []
        self.success_rate = []

    def run(self):
        while True:
            self.train_offline()
            if self.steps > self.num_steps:
                break

    def explore(self, state):
        # Act with randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float()/255.0
        
        with torch.no_grad():            
            state  = rand_crop(state, self.device)
            curr_state, _ = self.latent.encoder(state[:,:3,:,:])
            goal_state, _ = self.latent.encoder(state[:,3:,:,:])
            states = torch.cat((curr_state,goal_state),dim=1)
            action = self.policy.act(states)
        return action.item()
    
    def high_level_act(self, state):
        curr_state, goal_state = state['image'], state['goal']
        curr_state = torch.ByteTensor(
            cur_state).unsqueeze(0).to(self.device).float()/255.0
        goal_state = torch.ByteTensor(
            goal_state).unsqueeze(0).to(self.device).float()/255.0
        
        with torch.no_grad():            
            curr_state  = center_crop(curr_state, self.device)
            goal_state  = center_crop(goal_state, self.device)
            curr_state, _ = self.latent.encoder(curr_state)
            goal_state, _ = self.latent.encoder(goal_state)
            states = torch.cat((curr_state,goal_state),dim=1)
            action = self.policy.act(states)
        return action.view(-1).cpu().numpy()

    def train_offline(self):
        
        self.learn()

        self.steps += 1

        if self.steps % self.num_eval_steps == 0:
            self.evaluate()
            self.save_models(self.model_dir)

    def learn(self):
        if self.steps % self.target_update_interval == 0:
            self.update_target()

        # self.learn_latent()
        
        self.learn_sac()
        
        self.learn_dqn()
        
        self.learn_classifier()
  
    def learn_latent(self):
        images =\
            self.memory.sample_latent(self.batch_size)
        aug_images = get_aug_image(images, self.device) 
        latent_loss = self.calc_latent_loss(aug_images, aug_images)
        update_params(self.latent_optim, latent_loss)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/latent_loss', latent_loss.detach().item(),
                self.learning_steps)
            
    def learn_dqn(self):

        images, actions, rewards, next_images, not_dones =\
            self.memory.sample_discrete(self.batch_size)
        weights = 1.

        images = get_aug_image(images, self.device)
        next_images = get_aug_image(next_images, self.device)

        with torch.no_grad():
            features1, _ = self.latent.encoder(images[:,:3,:,:])
            features2, _ = self.latent.encoder(images[:,3:,:,:])
            features = torch.cat((features1,features2),dim=1)

            next_features1, _ = self.latent.encoder(next_images[:,:3,:,:])
            next_features2, _ = self.latent.encoder(next_images[:,3:,:,:])
            next_features = torch.cat((next_features1,next_features2),dim=1)

        states = features
        next_states = next_features
        
        dqn_loss = self.calc_dqn_loss(states, actions, rewards, next_states, not_dones, weights)

        update_params(self.dqn_optim, dqn_loss, retain_graph=True)

    def learn_sac(self):

        images, actions, rewards, next_images, not_dones =\
            self.memory.sample_discrete(self.batch_size)
        weights = 1.

        images = get_aug_image(images, self.device)
        next_images = get_aug_image(next_images, self.device)

        with torch.no_grad():
            features1, _ = self.latent.encoder(images[:,:3,:,:])
            features2, _ = self.latent.encoder(images[:,3:,:,:])
            features = torch.cat((features1,features2),dim=1)

            next_features1, _ = self.latent.encoder(next_images[:,:3,:,:])
            next_features2, _ = self.latent.encoder(next_images[:,3:,:,:])
            next_features = torch.cat((next_features1,next_features2),dim=1)

        states = features
        next_states = next_features

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(states, actions, rewards, next_states, not_dones, weights)
        policy_loss, entropies = self.calc_policy_loss(states, actions, rewards, next_states, not_dones, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss, retain_graph=True)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.use_per:
            self.memory.update_priority(errors)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)
            
    def learn_classifier(self):
    
        images, posb_actions, selc_actions =\
            self.memory.sample_classifier(self.batch_size)
        images = get_aug_image(images,self.device)

        with torch.no_grad():
            features1, _ = self.latent.encoder(images[:,:3,:,:])
            features2, _ = self.latent.encoder(images[:,3:,:,:])
            features = torch.cat((features1,features2),dim=1)

        states = features
        posb_actions = torch.cat((posb_actions,posb_actions,posb_actions,posb_actions),dim=0)
        selc_actions = torch.cat((selc_actions,selc_actions,selc_actions,selc_actions),dim=0)

        posb_classifier_loss = \
            self.calc_posb_classifier_loss(states, posb_actions)
        selc_classifier_loss = \
            self.calc_selc_classifier_loss(states, selc_actions)

        update_params(self.posb_classifier_optim, posb_classifier_loss, retain_graph=True)
        update_params(self.selc_classifier_optim, selc_classifier_loss, retain_graph=True)
        
        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/posb_classifier', posb_classifier_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/selc_classifier', selc_classifier_loss.detach().item(),
                self.learning_steps)

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def n_batch_mean(self, target, n=4):
        assert target.shape[0]%n ==0

        K = target.shape[0]//n
        values = 0
        for i in range(n):
            values += target[i*K:(i+1)*K]
        value = values/float(n)

        return value

    def calc_current_q(self, states, actions):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1, curr_q2 = self.n_batch_mean(curr_q1), self.n_batch_mean(curr_q2)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_current_q3(self, states, actions):
        curr_q1, curr_q2 = self.online_critic(states)		
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        # curr_q1, curr_q2 = self.n_batch_mean(curr_q1), self.n_batch_mean(curr_q2)
        return curr_q1, curr_q2

    def calc_target_q(self, rewards, next_states, not_dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

            next_q = self.n_batch_mean(next_q)

        assert rewards.shape == next_q.shape
        return rewards +  self.gamma_n * next_q # not_dones

    def calc_critic_loss(self, states, actions, rewards, next_states, not_dones, weights):

        curr_q1, curr_q2 = self.calc_current_q(states, actions)
        
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()


        # Max Q back up
        if self.max_q_backup:
            """when using max q backup"""
            next_actions_temp, _, _ = self.policy.sample(next_states.detach())
            target_qf1_values, target_qf2_values = self.calc_current_q3(next_states, next_actions_temp)
            target_qf1_values, target_qf2_values = self.n_batch_mean(target_qf1_values), self.n_batch_mean(target_qf2_values)
            target_qf1_values, target_qf2_values = target_qf1_values.max(1)[0].view(-1, 1), target_qf2_values.max(1)[0].view(-1, 1)
            
            next_q = torch.min(target_qf1_values, target_qf2_values)
            target_q = rewards + not_dones*self.gamma_n * next_q
        else:
            with torch.no_grad():
                target_q = self.calc_target_q(rewards, next_states, not_dones)       
        target_q = target_q.detach()

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)


        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        # add CQL 
        state_repeat = states.unsqueeze(1).repeat(1, self.num_random, 1).view(states.shape[0] * self.num_random, states.shape[1])
        next_state_repeat = next_states.unsqueeze(1).repeat(1, self.num_random, 1).view(states.shape[0] * self.num_random, states.shape[1])
        
        random_actions_tensor = torch.randint(self.num_action,(states.shape[0]*self.num_random, 1),dtype=torch.float).to(self.device)
        curr_actions_tensor, _, curr_log_pis = self.policy.sample(state_repeat)
        new_curr_actions_tensor, _, new_log_pis = self.policy.sample(next_state_repeat)
        q1_rand, q2_rand = self.calc_current_q3(state_repeat, random_actions_tensor)
        q1_curr_actions, q2_curr_actions = self.calc_current_q3(state_repeat, curr_actions_tensor)
        q1_next_actions, q2_next_actions = self.calc_current_q3(next_state_repeat, new_curr_actions_tensor)
        
        q1_rand, q2_rand = q1_rand.view(states.shape[0], self.num_random, 1), q2_rand.view(states.shape[0], self.num_random, 1)
        q1_curr_actions, q2_curr_actions = q1_curr_actions.view(states.shape[0], self.num_random, 1), q2_curr_actions.view(states.shape[0], self.num_random, 1)
        q1_next_actions, q2_next_actions = q1_next_actions.view(states.shape[0], self.num_random, 1), q2_next_actions.view(states.shape[0], self.num_random, 1)
        
        q1_rand, q2_rand = self.n_batch_mean(q1_rand), self.n_batch_mean(q2_rand)
        q1_next_actions, q2_next_actions = self.n_batch_mean(q1_next_actions), self.n_batch_mean(q2_next_actions)
        q1_curr_actions, q2_curr_actions = self.n_batch_mean(q1_curr_actions), self.n_batch_mean(q2_curr_actions)

        
        if self.min_q_version == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** self.num_action)#curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q1, cat_q2 = self.n_batch_mean(cat_q1), self.n_batch_mean(cat_q2)

        else:
            cat_q1 = torch.cat(
                [q1_rand,  curr_q1.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
            )
            cat_q2 = torch.cat(
                [q2_rand,  curr_q2.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
            )


        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
                    
        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - curr_q1.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - curr_q2.mean() * self.min_q_weight
        
        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        q1_loss = q1_loss + min_qf1_loss
        q2_loss = q2_loss + min_qf2_loss
        
        return q1_loss, q2_loss, errors, mean_q1, mean_q2
      
    def calc_dqn_loss(self, states, actions, rewards, next_states, not_dones, weights):
    
        curr_q = self.dqn(states)
        curr_q = self.n_batch_mean(curr_q)
        curr_q = curr_q.gather(1, actions.long())
        
        #with torch.no_grad():
        next_q = self.target_dqn(next_states)
        next_q = self.n_batch_mean(next_q)
        next_q = next_q.detach().max(1)[0].view(-1, 1)
        target_q = rewards + not_dones*self.gamma_n * next_q
        target_q = target_q.detach()
        
        q_loss = torch.mean((curr_q - target_q).pow(2) * weights)        

        # # add CQL 
        with torch.no_grad():
            state_repeat = states.unsqueeze(1).repeat(1, self.num_random, 1).view(states.shape[0] * self.num_random, states.shape[1])
            next_state_repeat = next_states.unsqueeze(1).repeat(1, self.num_random, 1).view(states.shape[0] * self.num_random, states.shape[1])
            
            random_actions_tensor = torch.randint(self.num_action,(states.shape[0]*self.num_random, 1),dtype=torch.float).to(self.device)
            curr_action_logits = self.dqn(state_repeat)
            curr_actions_tensor = torch.argmax(curr_action_logits, dim=1, keepdim=True)
            curr_actions_tensor = curr_actions_tensor.detach()
            next_action_logits = self.dqn(next_state_repeat)
            next_actions_tensor = torch.argmax(next_action_logits, dim=1, keepdim=True)
            next_actions_tensor = next_actions_tensor.detach()
            
        q_rand = self.dqn(state_repeat)
        q_curr = self.dqn(state_repeat)
        q_next = self.dqn(next_state_repeat)
        
        
        q_rand = q_rand.gather(1, random_actions_tensor.long())
        q_curr = q_curr.gather(1, curr_actions_tensor.long())
        q_next = q_next.gather(1, next_actions_tensor.long())
        
        q_rand = self.n_batch_mean(q_rand)
        q_curr = self.n_batch_mean(q_curr)
        q_next = self.n_batch_mean(q_next)
        
        cat_q = torch.cat(
            [q_rand, q_next, q_curr], 1
        )

        min_qf_loss = torch.logsumexp(cat_q / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
                    
        # """Subtract the log likelihood of data"""
        min_qf_loss = min_qf_loss - cat_q.mean() * self.min_q_weight
        
        q_loss = q_loss + min_qf_loss
        
        return q_loss

    def calc_policy_loss(self, states, actions, rewards, next_states, not_dones, weights):


        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies )).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def calc_latent_loss(self, aug_images, ori_images):

        features, dist = self.latent.encoder(aug_images[:,:3,:,:])
        dist_sample = dist.rsample()

        mu = dist.loc
        logvar = dist.scale

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        recon_img_dists = self.latent.decoder(
            dist_sample)

        log_likelihood_loss = recon_img_dists.log_prob(
            ori_images[:,:3,:,:]).mean(dim=0).sum()


        latent_loss =\
            self.beta*KLD - log_likelihood_loss # + 100*reconst_error

        if self.learning_steps % self.log_interval == 0:
            reconst_error = (
                ori_images[:,:3,:,:] - recon_img_dists.loc
                ).pow(2).mean(dim=(0, 1)).sum().item()
            self.writer.add_scalar(
                'stats/reconst_error', reconst_error, self.learning_steps)

        return latent_loss
    
    def calc_posb_classifier_loss(self, states, posb_action):
        
        pred_act = self.posb_classifier(states)        
        # kk = self.crossEnt(pred_act,posb_action)
        classifier_loss = torch.mean((pred_act - posb_action).pow(2))  
        
        return classifier_loss
    
    def calc_selc_classifier_loss(self, states, selc_action):
        
        pred_act = self.selc_classifier(states)        
        # kk = self.crossEnt(pred_act,posb_action)
        classifier_loss = torch.mean((pred_act - selc_action).pow(2))  
        
        return classifier_loss
    
    def evaluate(self):
        
        pred_suc, posb_suc, selc_suc, dqn_suc = 0, 0, 0, 0
        mean_q1, mean_q2 = 0, 0
        loop = 10
        for _ in range(loop):
            images, gt_actions, gt_posb_action =\
                self.val_memory.sample_evaluate(1,k=1)
                
            with torch.no_grad():
                images  = rand_crop(images, self.device)
                curr_state, _ = self.latent.encoder(images[:,:3,:,:])
                goal_state, _ = self.latent.encoder(images[:,3:,:,:])
                states = torch.cat((curr_state,goal_state),dim=1)
                
                pred_action = self.policy.act(states)
                posb_action = self.posb_classifier.act(states)                
                selc_action = self.selc_classifier.act(states)
                dqn_action = self.dqn.act(states)
                
                curr_q1, curr_q2 = self.calc_current_q3(states, pred_action)
                gt_curr_q1, gt_curr_q2 = self.calc_current_q3(states, gt_actions)
                
                mean_q1 += gt_curr_q1.detach().mean().item()/loop
                mean_q2 += gt_curr_q2.detach().mean().item()/loop
    
            gtq = np.round(torch.min(gt_curr_q1,gt_curr_q2).view(-1).cpu().numpy(),3)
            predq = np.round(torch.min(curr_q1,curr_q2).view(-1).cpu().numpy(),3)
            

            gt_posb_action = gt_posb_action.cpu().numpy()
            gt_actions = gt_actions.view(-1).cpu().numpy().astype(np.uint8)
            pred_action = pred_action.view(-1).cpu().numpy()
            posb_action = posb_action.view(-1).cpu().numpy()
            selc_action = selc_action.view(-1).cpu().numpy()
            dqn_action = dqn_action.view(-1).cpu().numpy()
            
            for i in range(11):            
                pos = list(np.where(gt_posb_action[i]>0.1)[0])
                if pred_action[i] in pos:
                    pred_suc += 1
                if posb_action[i] in pos:
                    posb_suc += 1
                if selc_action[i] in pos:
                    selc_suc += 1
                if dqn_action[i] in pos:
                    dqn_suc += 1
                    
        self.success_rate.append([pred_suc,posb_suc,selc_suc,dqn_suc])
        
        print()
        print(f"eval {self.steps}")
        print(f"gtq  values :")
        print(f"{gtq[:5]}")
        print(f"{gtq[5:]}")
        print(f"  q  values :")
        print(f"{predq[:5]}")
        print(f"{predq[5:]}")
        print(f"Is right direction?")
        print(f"{(gtq>=predq)[:5]}")
        print(f"{(gtq>=predq)[5:]}")
        print(f" gt  action: {gt_actions}")
        print(f"pred action: {pred_action}, success: {round(pred_suc/(loop*self.num_action),3)}")
        print(f"posb action: {posb_action}, success: {round(posb_suc/(loop*self.num_action),3)}")
        print(f"selc action: {selc_action}, success: {round(selc_suc/(loop*self.num_action),3)}")
        print(f"dqn  action: {dqn_action}, success: {round(dqn_suc/(loop*self.num_action),3)}")
        
        np.save("data/success_rate.npy", self.success_rate)
        
        
        self.cur_mean_q = min(mean_q1,mean_q2)
        self.cur_q_val_history.append(self.cur_mean_q)
        temp1 = np.array(self.cur_q_val_history)
        np.save('data/cur_q_val_history2.npy',temp1)
        
        if self.cur_mean_q > self.highest_q and self.steps > 4999:
            self.highest_q = self.cur_mean_q
            self.save_models(self.model_dir)
            self.saved_step.append(self.steps)
            temp2 = np.array(self.saved_step)
            np.save('data/saved_list2.npy',temp2)
            
        if self.steps % 10000 == 0:
            self.save_models(self.model_dir, self.steps)
                   
    def test(self):
        
        self.load_models()
        
        print()
        print(f"Steps {self.steps} Evaluation start")
        episodes = 50
        num_success = 0
        for epi in range(episodes):

            all_success = True
            ###################### High level #########################
            high_d = False
            high_steps = 0
            high_o = self.env.reset()

            while not high_d:
                
                
                # High policy
                pred_task = self.high_level_act(high_o)
                pred_task = int(pred_task)
                print(f'pred task: {pred_task}')
                pos_task = self.env.get_possible_action()
                
                task_num = self.env.select_task(pred_task)
                print(f'pos  task: {pos_task}')
                # task_num = pos_task[random.randint(0, len(pos_task)-1)]
                if pred_task not in pos_task:
                    all_success = False
                    

                ##################### Low level #########################
                # low_d = False
                # low_return = 0.0
                # low_steps = 0
                # low_o,_, _  = self.env.get_low_state()

                # while not low_d and low_steps <30 and task_num != 14:
                    
                #     # Low policy
                #     # if low_steps % 6 == 0:
                #     #     low_a, q1, q2 = self.test_action(low_o['image'], low_o['hybrid'], low_o['task'], q_value=True)
                #     #     print(f"Q1: {np.round(q1,3)}, Q2: {np.round(q2,3)}")
                #     # else:
                #     #     low_a = self.test_action(low_o['image'], low_o['hybrid'], low_o['task'])
                #     low_next_o, low_r, low_d = self.env.low_step_script(task_num=task_num) 

                #     if low_d:
                #         print("OH YEAH")
                #     low_steps +=1
                #     low_o = low_next_o
                #     low_return += low_r

                print(f"Task {task_num:>2} {self.action_dict[str(task_num)]:<20}")#" reward: {low_return:<3}")

                ##################### Low level #########################
                time.sleep(0.5)
                high_next_o, high_r, high_d =  self.env.high_step(task_num)
                
                c = high_o['image']
                n = high_next_o['image']
                g = high_next_o['goal']
                c, n, g = np.transpose(c,[1,2,0]), np.transpose(n,[1,2,0]), np.transpose(g,[1,2,0])
                cat_img = np.concatenate((c,n,g),axis=1)
                plt.imsave(f'check/test/{epi}_step{high_steps}_a{pred_task}.png',cat_img)
                
                high_o = high_next_o
                high_steps += 1

            if high_d:
                print("All done")
            if all_success:
                num_success += 1
            
            print(f'{num_success}/{episodes}')
                
            ###################### High level #########################

    def load_models(self):
        self.policy.load('model/policy.pth', self.device)
        self.latent.load('model/latent_base.pth', self.device)
        self.online_critic.load('model/online_critic.pth', self.device)
        self.target_critic.load('model/target_critic.pth', self.device)

    def save_models(self, save_dir, num=None):
        if num is not None:
            self.policy.save(os.path.join(save_dir, 'policy.pth'))
            self.latent.save(os.path.join(save_dir, 'latent.pth'))
            self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
            self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))
        else:
            self.policy.save(os.path.join(save_dir, f'policy_{num}.pth'))
            self.latent.save(os.path.join(save_dir, f'latent_{num}.pth'))
            self.online_critic.save(os.path.join(save_dir, f'online_critic_{num}.pth'))
            self.target_critic.save(os.path.join(save_dir, f'target_critic_{num}.pth'))

    def load_data(self,num_rollout):

        data_path_list = []
        for i in range(1,num_rollout-3):
            data_path_list.append(f"../rollout_R/high_level/rollout{i}.npy")
        self.memory.add_data_to_buffer(data_path_list, last_stage=True)
        data_path_list = []
        for i in range(num_rollout-3,num_rollout):
            data_path_list.append(f"../rollout_R/high_level/rollout{i}.npy")
        self.val_memory.add_data_to_buffer(data_path_list, last_stage=True)
        print(f"High level data is loaded")


    def seed_all(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
