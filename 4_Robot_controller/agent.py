import os
import time
import random
import numpy as np
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision.utils import save_image

import matplotlib.pyplot as plt

from memory import Memory
from network import GaussianPolicy, DeterministicPolicy, TwinnedQNetwork, LatentNetwork, TaskNetwork
from utils import grad_false, hard_update, soft_update, update_params, rand_crop, get_aug_image


class Agent:
    def __init__(
        self,
        env,
        log_dir,
        env_type='custom',
        num_steps=3000000,
        init_latent=True,
        initial_latent_steps=100000,
        batch_size=256,
        latent_batch_size=32,
        beta=1e-4,
        policy_lr=1e-4,
        critic_lr=3e-4,
        latent_lr=0.0001,
        latent_dim=256,
        hidden_units=[256, 256],
        memory_size=1e5,
        gamma=0.99,
        target_update_interval=1,
        tau=0.005,
        entropy_tuning=True,
        ent_coef=0.2,
        leaky_slope=0.2,
        grad_clip=None,
        updates_per_step=1,
        learning_log_interval=100,
        eval_interval=20000,
        cuda=True,
        seed=0,
        gpu=0,
        num_rollout=2,
        dataset_type='expert',
        use_cql=True,
		use_bc=True,
        task_ratio=True,
        reward_ratio=None,
		policy_type='Stochastic'):

        self.seed = seed        
        self.use_bc = use_bc
        self.use_cql = use_cql
        self.policy_type = policy_type
        self.dataset_type = dataset_type

        # Dataset_setting
        self.task_ratio = task_ratio
        self.reward_ratio = reward_ratio
        self.num_step_after_done = 3 
        
        # Date shape
        self.env = env
        self.observation_shape = self.env.low_observation_space.shape
        self.hybrid_state_shape = self.env.hybrid_state_space.shape
        self.high_action_shape = self.env.high_action_space.shape
        self.low_action_shape = self.env.low_action_space.shape
        
        # Basic setting
        self.seed_all(seed)
        self.action_define()
        self.directory_setting(log_dir)
                
        # Set device
        self.device = torch.device(
            "cuda:{}".format(gpu) if cuda and torch.cuda.is_available() else "cpu")
        print(f"CUDA available: {torch.cuda.is_available()}, device: {self.device}") 

        # Set network
        if self.policy_type == 'Stochastic':
            self.policy = GaussianPolicy(
                latent_dim + self.hybrid_state_shape[0] + self.high_action_shape[0],
                self.low_action_shape[0], hidden_units).to(self.device)
            # Auto entropy tuning
            if entropy_tuning:
                # Target entropy is -|A|.
                self.target_entropy = -torch.prod(
                    torch.Tensor(self.low_action_shape)).item()
                # We optimize log(alpha), instead of alpha.
                self.log_alpha = torch.zeros(
                    1, requires_grad=True, device=self.device)
                self.alpha = self.log_alpha.exp()
                self.alpha_optim = Adam([self.log_alpha], lr=critic_lr)
            else:
                self.alpha = torch.tensor(ent_coef).to(self.device)            
            self.entropy_tuning = entropy_tuning
        else:
            self.policy = DeterministicPolicy(
                latent_dim + self.hybrid_state_shape[0] + self.high_action_shape[0],
                self.low_action_shape[0], hidden_units).to(self.device)
            self.entropy_tuning = False
        self.critic = TwinnedQNetwork(latent_dim + self.hybrid_state_shape[0] + self.high_action_shape[0],
            self.low_action_shape[0], hidden_units).to(self.device)
        self.critic_target = TwinnedQNetwork(latent_dim + self.hybrid_state_shape[0] + self.high_action_shape[0],
            self.low_action_shape[0], hidden_units).to(self.device).eval()
        self.latent = LatentNetwork(
            self.observation_shape[0], latent_dim, leaky_slope, hidden_units=[256,256]
            ).to(self.device)
        
        # Policy is updated without the encoder.
        self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=critic_lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=critic_lr)
        # self.latent_optim = Adam(self.latent.parameters(), lr=latent_lr)

        # Loss
        self.MseLoss = nn.MSELoss()

        # Offline dataset setting        
        self.memory = LazyMemory(
            memory_size, self.observation_shape, self.hybrid_state_shape, self.low_action_shape, self.high_action_shape,
            self.reward_ratio, self.task_ratio, self.num_step_after_done, batch_size, self.device)
        self.load_data(num_rollout) 

        # Copy parameters of the learning network to the target network.
        hard_update(self.critic_target, self.critic)
        # Disable gradient calculations of the target network.
        grad_false(self.critic_target)

        # CQL-setting
        self.max_q_backup = True
        self.num_random = 10
        if self.use_cql:            
            self.temp = 1.0
            self.min_q_weight = 1.0
            self.min_q_version = 2            

            self.with_lagrange = True
            if self.with_lagrange:
                self.target_action_gap = 5.0
                self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_prime_optimizer = Adam(
                        [self.log_alpha_prime],
                        lr=3e-4,)        
        # BC-setting
        if self.use_bc:
            self.bc_coef = 2.5

        self.beta = beta
        self.init_latent = init_latent
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.initial_latent_steps = initial_latent_steps
        self.num_steps = num_steps
        self.tau = tau
        self.batch_size = batch_size
        self.latent_batch_size = latent_batch_size
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.learning_log_interval = learning_log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

        if not self.init_latent:
            self.latent.load('model/final_latent.pth', self.device)
            print("VAE loaded")

    def run(self):
        while True:
            self.train_offline()
            if self.steps > self.num_steps:
                break

    def deque_to_batch(self, state, robot_state, task):
        # Convert deques to batched tensor.
        state = np.array(state, dtype=np.uint8)
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float()/255.0
        robot_state = torch.FloatTensor(
            robot_state).unsqueeze(0).to(self.device)
        task = torch.FloatTensor(
            task).unsqueeze(0).to(self.device)
        with torch.no_grad():

            state = rand_crop(state, self.device)

            feature, _ = self.latent.encoder(state)
            feature = feature.view(1, -1)

        feature_state = torch.cat([feature, robot_state, task], dim=-1)
        return feature_state

    def explore(self, state, robot_state, task):
        feature_state = self.deque_to_batch(state, robot_state, task)
        
        with torch.no_grad():
            action, _, _ = self.policy.sample(feature_state)
        return action.cpu().numpy().reshape(-1)

    def test_action(self, state, robot_state, task, q_value=False):
        feature_state = self.deque_to_batch(state, robot_state, task)
        
        with torch.no_grad():
            _, _, action = self.policy.sample(feature_state)
        if q_value:
            with torch.no_grad():
                q1, q2 = self.critic.forward(feature_state, action)
                q1, q2 = q1.cpu().numpy().reshape(-1), q2.cpu().numpy().reshape(-1)
            return action.cpu().numpy().reshape(-1), q1, q2
        else:
            return action.cpu().numpy().reshape(-1)

    def train_offline(self):
        
        if self.init_latent:
            if self.learning_steps < self.initial_latent_steps:
               print('-'*60)
               print('Learning the latent model only...')
               for _ in range(self.initial_latent_steps):
                   self.learning_steps += 1
                   if self.learning_steps % 1000 == 0:
                       print(self.learning_steps)
                   self.learn_latent()
               print('Finish learning the latent model.')
               print('-'*60)
            self.save_latent()
            print("save_latent")

        for _ in range(self.updates_per_step):
            self.learn()

        self.steps += 1

        if self.steps % self.eval_interval == 0:
            self.evaluate()
            # self.save_models()
        if self.steps % 10000==0:
            self.save_models()
        if self.steps % 50000 == 0:
            self.save_models(self.steps)

    def learn(self):        
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # Update the latent model.
        # self.learn_latent()
        # Update policy and critic.
        self.learn_rl()

        self.learning_steps += 1

    def learn_latent(self):
        images =\
            self.memory.sample_latent(self.latent_batch_size)

        aug_images, gt_images = get_aug_image(images, self.device)

        latent_loss = self.calc_latent_loss(aug_images, gt_images)
        update_params(
            self.latent_optim, self.latent, latent_loss, self.grad_clip)

        if self.learning_steps % self.learning_log_interval == 0:
            self.writer.add_scalar(
                'loss/latent_loss', latent_loss.detach().item(),
                self.learning_steps)

    def learn_rl(self):

        if self.reward_ratio is None:
            images, robot_states, actions, rewards, next_images, next_robot_states, not_dones, tasks =\
                self.memory.sample_sac(self.batch_size)
        else:
            images, robot_states, actions, rewards, next_images, next_robot_states, not_dones, tasks =\
                self.memory.sample_reward_ratio(self.batch_size)     


        # NOTE: Don't update the encoder part of the policy here.
        with torch.no_grad():
            images = get_aug_image(images, self.device)
            features, _ = self.latent.encoder(images)

            next_images = get_aug_image(next_images, self.device)
            next_features, _ = self.latent.encoder(next_images)

        robot_states_repeat = torch.cat((robot_states,robot_states,robot_states,robot_states),dim=0)
        next_robot_states_repeat = torch.cat((next_robot_states,next_robot_states,next_robot_states,next_robot_states),dim=0)
        tasks_repeat = torch.cat((tasks,tasks,tasks,tasks),dim=0)
        actions_repeat = torch.cat((actions,actions,actions,actions),dim=0)

        states = torch.cat((features, robot_states_repeat,tasks_repeat), dim=1)
        next_states = torch.cat((next_features, next_robot_states_repeat,tasks_repeat), dim=1)
        
        policy_loss, entropies, q1_loss, q2_loss = self.calc_offline_rl_loss(states, actions_repeat, rewards, next_states, not_dones)        
        
        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip, retain_graph=True)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip, retain_graph=True)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip, retain_graph=True)

        if self.learning_steps % self.learning_log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            if self.policy_type == 'Stochastic':
                self.writer.add_scalar(
                    'stats/alpha', self.alpha.detach().item(),
                    self.learning_steps)
                self.writer.add_scalar(
                    'stats/entropy', entropies.detach().mean().item(),
                    self.learning_steps)

    def calc_latent_loss(self, aug_images, gt_images):

        _, dist = self.latent.encoder(aug_images)
        dist_sample = dist.rsample()

        mu = dist.loc
        logvar = dist.scale

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        recon_img_dists = self.latent.decoder(
            dist_sample)

        log_likelihood_loss = recon_img_dists.log_prob(
            gt_images).mean(dim=0).sum()

        latent_loss =\
            self.beta*KLD - log_likelihood_loss

        if self.learning_steps % self.learning_log_interval == 0:
            reconst_error = (
                gt_images - recon_img_dists.loc
                ).pow(2).mean(dim=(0, 1)).sum().item()
            self.writer.add_scalar(
                'stats/reconst_error', reconst_error, self.learning_steps)

        return latent_loss
    
    def calc_offline_rl_loss(self, states, actions, rewards, next_states, not_dones):

        """
        Policy and Alpha Loss
        """
        # with torch.autograd.set_detect_anomaly(True):
        sampled_actions, entropies, _ = self.policy.sample(states.detach())

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
        else:
            entropy_loss = 0.
            
        q1, q2 = self.critic(states, sampled_actions)
        q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy). 
        if self.policy_type == 'Stochastic':
            reinp_loss = torch.mean((- q - self.alpha * entropies))
        else:
            reinp_loss = torch.mean((- q))
        
        policy_loss = reinp_loss
        
        if self.use_bc:
            # norm =  (q + self.alpha * entropies) if self.policy_type == 'Stochastic' else q
            lmbda =  1.0 #self.bc_coef/norm.abs().mean().detach() #
            bc_loss = self.MseLoss(sampled_actions, actions)        
            policy_loss = lmbda*reinp_loss + bc_loss

        """
        QF Loss
        """
        q1_pred, q2_pred = self.critic(states, actions)
        q1_pred, q2_pred = self.n_batch_mean(q1_pred), self.n_batch_mean(q2_pred)
        # E[Q(z(t+1), a(t+1)) + alpha * H(pi)]

        with torch.no_grad():
            if self.max_q_backup:
                """when using max q backup"""
                if self.policy_type == 'Stochastic':
                    next_actions, _, _ = self.policy.sample_repeat(next_states.detach(), num_repeat=self.num_random)
                else:
                    _, _, next_actions = self.policy.sample_repeat(next_states.detach(), num_repeat=self.num_random)
                target_qf1_value, target_qf2_value = self.critic_target.forward_repeat(next_states, next_actions, num_repeat=self.num_random)
                target_qf1_value, target_qf2_value = self.n_batch_mean(target_qf1_value), self.n_batch_mean(target_qf2_value)
                target_qf1_value, target_qf2_value = target_qf1_value.max(1)[0].view(-1, 1), target_qf2_value.max(1)[0].view(-1, 1)
                next_q = torch.min(target_qf1_value, target_qf2_value)
            else:
                if self.policy_type == 'Stochastic':
                    next_actions, next_entropies, _ = self.policy.sample(next_states.detach())
                else:
                    _, _, next_actions = self.policy.sample(next_states.detach())
                next_q1, next_q2 = self.critic_target(next_states, next_actions)
                next_q1, next_q2 =  self.n_batch_mean(next_q1), self.n_batch_mean(next_q2)
                if self.policy_type == 'Stochastic':
                    next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies
                else:
                    next_q = torch.min(next_q1, next_q2)

        target_q = rewards + not_dones * self.gamma * next_q
        target_q = target_q.detach()

        # Critic losses are mean squared TD errors.
        q1_loss = self.MseLoss(q1_pred, target_q)
        q2_loss = self.MseLoss(q2_pred, target_q)

        # add CQL
        if self.use_cql:
            random_actions_tensor = torch.empty((q2_pred.shape[0] *4 * self.num_random, actions.shape[-1]), device=self.device).uniform_(-1, 1) # .cuda()        
            curr_actions_tensor, curr_entropies, _ = self.policy.sample_repeat(states.detach(), num_repeat=self.num_random)
            new_curr_actions_tensor, new_entropies, _ = self.policy.sample_repeat(next_states.detach(), num_repeat=self.num_random)
            q1_rand, q2_rand = self.critic.forward_repeat(states, random_actions_tensor, num_repeat=self.num_random)
            q1_curr_actions, q2_curr_actions = self.critic.forward_repeat(states, curr_actions_tensor, num_repeat=self.num_random)
            q1_next_actions, q2_next_actions = self.critic.forward_repeat(states, new_curr_actions_tensor, num_repeat=self.num_random)
            
            if self.min_q_version == 3:
                # importance sammpled version
                random_entropies = -np.log(0.5 ** curr_actions_tensor.shape[-1])
                cat_q1 = torch.cat(
                    [q1_rand + random_entropies, q1_next_actions + new_entropies.detach(), q1_curr_actions + curr_entropies.detach()], 1
                )
                cat_q2 = torch.cat(
                    [q2_rand + random_entropies, q2_next_actions + new_entropies.detach(), q2_curr_actions + curr_entropies.detach()], 1
                )
                cat_q1, cat_q2 = self.n_batch_mean(cat_q1), self.n_batch_mean(cat_q2)
            else:
                q1_rand, q2_rand = self.n_batch_mean(q1_rand), self.n_batch_mean(q2_rand)
                q1_curr_actions, q2_curr_actions = self.n_batch_mean(q1_curr_actions), self.n_batch_mean(q2_curr_actions)
                q1_next_actions, q2_next_actions = self.n_batch_mean(q1_next_actions), self.n_batch_mean(q2_next_actions)
                cat_q1 = torch.cat(
                    [q1_rand, q1_next_actions,
                    q1_curr_actions], 1
                )
                cat_q2 = torch.cat(
                    [q2_rand, q2_next_actions,
                    q2_curr_actions], 1
                )
            
            min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
            min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
                        
            """Subtract the log likelihood of data"""
            min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
            min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight
            
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
        
        if self.learning_steps % self.learning_log_interval == 0:
            mean_q1, mean_q2 = q1_pred.detach().mean().item(), q2_pred.detach().mean().item()
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'loss/reinp_loss', reinp_loss.detach().item(),
                self.learning_steps)
            if self.policy_type=='Stochastic':
                self.writer.add_scalar(
                    'loss/alpha', entropy_loss.detach().item(),
                    self.learning_steps)
            if self.use_cql:                
                rand_q1, rand_q2 = q1_rand.detach().mean().item(), q2_rand.detach().mean().item()
                next_q1, next_q2 = q1_next_actions.detach().mean().item(), q2_next_actions.detach().mean().item()
                curr_q1, curr_q2 = q1_curr_actions.detach().mean().item(), q2_curr_actions.detach().mean().item()
                self.writer.add_scalar(
                    'stats/rand_Q1', rand_q1, self.learning_steps)
                self.writer.add_scalar(
                    'stats/rand_Q2', rand_q2, self.learning_steps)
                self.writer.add_scalar(
                    'stats/next_Q1', next_q1, self.learning_steps)
                self.writer.add_scalar(
                    'stats/next_Q2', next_q2, self.learning_steps)
                self.writer.add_scalar(
                    'stats/curr_Q1', curr_q1, self.learning_steps)
                self.writer.add_scalar(
                    'stats/curr_Q2', curr_q2, self.learning_steps)
            if self.use_bc:
                self.writer.add_scalar(
                    'loss/bc_loss', bc_loss.detach().item(),
                    self.learning_steps)
                self.writer.add_scalar(
                    'stats/lmbda', lmbda,
                    self.learning_steps)

        return policy_loss, entropies, q1_loss, q2_loss

    def calc_entropy_loss(self, entropies):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies).detach())
        return entropy_loss

    def n_batch_mean(self, target, n=4):
        assert target.shape[0]%n ==0

        K = target.shape[0]//n
        values = 0
        for i in range(n):
            values += target[i*K:(i+1)*K]
        value = values/float(n)

        return value

    def evaluate(self):
        print()
        print(f"Steps {self.steps} Evaluation start")
        episodes = 3
        task_list = []
        success_list = []
        for epi in range(episodes):            
            
            ###################### High level #########################
            high_d = False
            high_steps = 0
            high_o = self.env.reset(epi)

            while not high_d:
                pos_task = self.env.get_possible_action()
                # High policy
                task_num = pos_task[random.randint(0, len(pos_task)-1)]
                ###################### Low level #########################
                low_d = False
                low_return = 0.0
                low_steps = 0
                low_o,_, _  = self.env.get_low_state()

                if task_num != 10:
                    task_list.append(task_num)
                print(f"Pos task {pos_task}, Select {task_num:>2} {self.action_dict[str(task_num)]:<20}")
                while not low_d and low_steps <40 and task_num != 10:
                    
                    if low_steps % 8 == 0:
                        low_a, q1, q2 = self.test_action(low_o['image'], low_o['hybrid'], low_o['task'], q_value=True)
                        print(f"Q1: {np.round(q1,3)}, Q2: {np.round(q2,3)}, Act: {np.round(low_a,3)}")
                    else:
                        low_a = self.test_action(low_o['image'], low_o['hybrid'], low_o['task'])
                    low_next_o, low_r, low_d = self.env.low_step(task_num=task_num,low_a=low_a) 

                    if low_d:
                        print("                 OH YEAH")
                    low_steps +=1
                    low_o = low_next_o
                    low_return += low_r

                if task_num != 10:
                    if low_return:
                        success_list.append(1)
                    else:
                        success_list.append(0)

                ###################### Low level #########################

                high_next_o, high_r, high_d =  self.env.high_step(task_num)

                high_o = high_next_o
                high_steps += 1

                if high_d:
                    print("All done")

            ###################### High level #########################
            self.evaluate_info.append([self.steps, task_list,success_list, (sum(success_list)/len(success_list))])
        
        np.save(os.path.join(self.data_dir ,'evaluate.npy'), np.array(self.evaluate_info))
        self.cal_task_q_value(save=True)

        print(f"Steps {self.steps} Evaluation done")
        
    def cal_task_q_value(self, save=False):        
        
        all_q_value = 0
        roop = 3
        for task_id in range(self.high_action_shape[0]-1):
            cur_task_q_val = 0
            
            for _ in range(roop):
                images, robot_states, actions, _, _, _, _, tasks =\
                    self.memory.sample_task(self.batch_size, task_id)
                
                # NOTE: Don't update the encoder part of the policy here.
                with torch.no_grad():
                    images = rand_crop(images, self.device) 
                    features, _ = self.latent.encoder(images)
                    
                    states = torch.cat((features, robot_states, tasks), dim=1)
                    curr_q1, curr_q2 = self.critic(states, actions)
                    curr_q1, curr_q2 = curr_q1.detach().mean().item(), curr_q2.detach().mean().item()
                    cur_task_q_val += min(curr_q1, curr_q2)/roop

            self.cur_q_val_history[task_id].append(cur_task_q_val)
            
            all_q_value += cur_task_q_val/self.high_action_shape[0]

        if save:
            temp1 = np.array(self.cur_q_val_history)
            np.save(os.path.join(self.data_dir ,'cur_q_val_history.npy'),temp1)
            
            if all_q_value > self.highest_q and self.steps > 9999:
                self.highest_q = all_q_value
                self.save_models(self.steps)
                print("Model saved")
                self.saved_step.append(self.steps)
                temp2 = np.array(self.saved_step)
                np.save(os.path.join(self.data_dir, 'saved_list.npy'),temp2)
     
    def test(self):
        self.load_models()
        print()
        print(f"Steps {self.steps} Evaluation start")
        episodes = 2
        for _ in range(episodes):

            ###################### High level #########################
            high_d = False
            high_steps = 0
            high_o = self.env.reset()

            while not high_d:
                pos_task = self.env.get_possible_action()
                # High policy
                task_num = pos_task[random.randint(0, len(pos_task)-1)]
                ###################### Low level #########################
                low_d = False
                low_return = 0.0
                low_steps = 0
                low_o,_, _  = self.env.get_low_state()

                print(f"Task {task_num:>2} {self.action_dict[str(task_num)]:<20}")
                while not low_d and low_steps <40 and task_num != 10:
                    
                    if low_steps % 7 == 0:
                        low_a, q1, q2 = self.test_action(low_o['image'], low_o['hybrid'], low_o['task'], q_value=True)
                        print(f"Q1: {np.round(q1,3)}, Q2: {np.round(q2,3)}, action: {np.round(low_a,3)}")
                    else:
                        low_a = self.test_action(low_o['image'], low_o['hybrid'], low_o['task'])
                    low_next_o, low_r, low_d = self.env.low_step(task_num=task_num,low_a=low_a) 

                    if low_d:
                        print("                 OH YEAH")
                    low_steps +=1
                    low_o = low_next_o
                    low_return += low_r

                print(f"reward: {low_return:<3}")

                ###################### Low level #########################

                high_next_o, high_r, high_d =  self.env.high_step(task_num)

                high_o = high_next_o
                high_steps += 1

                if high_d:
                    print("All done")

            ###################### High level #########################

    def save_models(self, num=None):
        if num is None:
            # self.latent.save(os.path.join(self.model_dir, 'latent.pth'))
            self.policy.save(os.path.join(self.model_dir, 'policy.pth'))
            self.critic.save(os.path.join(self.model_dir, 'critic.pth'))
            self.critic_target.save(
                os.path.join(self.model_dir, 'critic_target.pth'))
        else:
            # self.latent.save(os.path.join(self.model_dir, f'latent_{num}.pth'))
            self.policy.save(os.path.join(self.model_dir, f'policy_{num}.pth'))
            self.critic.save(os.path.join(self.model_dir, f'critic_{num}.pth'))
            self.critic_target.save(
                os.path.join(self.model_dir, f'critic_target_{num}.pth'))

    def save_latent(self):
        self.latent.save('latent.pth')

    def load_models(self):
        self.latent.load('model/latent_base.pth', self.device)
        self.policy.load('model/policy.pth', self.device)
        self.critic.load('model/critic.pth', self.device)
        self.critic_target.load(
            'model/critic_target.pth', self.device)
        print("All models are loaded")

    def load_data(self, num_rollout):

        if self.dataset_type == 'expert':
            data_path = "../rollout_R/"
        else:
            data_path = "../rollout_F/"
        print(data_path)
        print(f"Num rollout: {num_rollout}")
        for task in range(self.high_action_shape[0]-1):
            data_path_list = []
            for roll in range(1,num_rollout):
                data_path_list.append(data_path+f"task{task}/rollout{roll}.npy")
            self.memory.add_data_to_buffer(data_path_list)
            print(f"Task {task} data is loaded")
        data_path_list = []
        self.memory.cal_sampling_ratio()

    def directory_setting(self, log_dir):
        
        if self.dataset_type=='expert':
            log_dir = log_dir + '_dR'
        else:
            log_dir = log_dir + '_dF'
        if self.use_cql:
            log_dir = log_dir + '_cql'
        if self.use_bc:
            log_dir = log_dir + '_bc' 
        if self.task_ratio:
            log_dir = log_dir + '_tr'
        log_dir = log_dir + f'_{self.policy_type[:3]}_{datetime.now().strftime("%m%d-%H%M")}_{self.seed}'
        
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.data_dir = os.path.join(log_dir, 'data')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.writer = SummaryWriter(log_dir=self.summary_dir)

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
        self.task_highest_q = [-np.inf] * self.high_action_shape[0]
        self.cur_q_val_history = []
        for _ in range(self.high_action_shape[0]):
            self.cur_q_val_history.append([])
        self.saved_step = []
        self.evaluate_info = []

    def seed_all(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

    def __del__(self):
        self.writer.close()
        self.env.close()
