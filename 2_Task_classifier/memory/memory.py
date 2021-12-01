from collections import deque
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

class Memory(dict):
    ''' Memory is memory-efficient but time-inefficient. '''
    keys = ['state', 'action', 'posb_action', 'selc_action', 'reward', 'done', 'next_state',]

    def __init__(self, capacity, observation_shape, action_shape, device):
        super(LazyMemory, self).__init__()
        self.capacity = int(1)
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = device
        self.num_action=11

        self.reward_ind = []
        self.action_ind = []

        for i in range(self.num_action):
            self.action_ind.append([])


        self.reset()

    def reset(self):
        self.is_set_init = False
        self._p = 0
        self._n = 0
        self._s = 0
        for key in self.keys:
            self[key] = [None] * self.capacity
        print("RESET")

    def get_data(self, data_path_list):
        buffer_size = 0
        data_list = []
        for path in data_path_list:
            data = np.load(path, allow_pickle=True)
            data_list.append(data)
            buffer_size += self.get_buffer_size(data)
            data = []

        return data_list, buffer_size

    def get_buffer_size(self, data):
        num_transitions = 0
        for i in range(len(data)):
            for j in range(len(data[i]['observations'])):
                num_transitions += 1
        return num_transitions

    def add_data_to_buffer(self, data_path_list, last_stage=False):
        data_list, buffer_size = self.get_data(data_path_list)
        self.capacity += buffer_size

        k = 0
        for data in data_list:
            k += 1
            for j in range(len(data)):
                assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
                    data[j]['next_observations']))

                bf_r = 0
                for i in range(len(data[j]['actions'])):

                    cur_state = data[j]['observations'][i]['image'][:3,:,:]
                    goal_state = data[j]['observations'][i]['goal'][:3,:,:]   
                    
                    state = np.concatenate((cur_state,goal_state),axis=0)
                    action =np.where(data[j]['actions'][i]==1)[0]
                    
                    posb_action = np.zeros((self.action_shape,))
                    posb_action[data[j]['possible_action'][i]] = 1.0/len(data[j]['possible_action'][i])
                    
                    selc_action = data[j]['actions'][i]

                    if last_stage == True:
                        not_done = [not(data[j]['terminals'][i])]
                        reward = data[j]['rewards'][i]
                    else:
                        not_done = [True]
                        reward = [0]

                    if reward != bf_r:
                        self.reward_ind.append(self._p)
                    bf_r = reward
                    
                    self.action_ind[action[0]].append(self._p)

                    if self._n < self.capacity:
                        for key in self.keys:
                            self[key] += [None]                       


                    next_cur_state = data[j]['next_observations'][i]['image'][:3,:,:]
                    next_goal_state = data[j]['next_observations'][i]['goal'][:3,:,:]
                    next_state = np.concatenate((next_cur_state,next_goal_state),axis=0)

                    self.append(state, action, posb_action, selc_action, reward, not_done, next_state)

        print(f"capacity : {self.capacity}")
        print(f"p : {self._p}")
        print(f"n : {self._n}")
        data_list=[]

    def append(self, state, action, posb_action, selc_action, reward, done, next_state):
        self['state'][self._p] = state
        self['action'][self._p] = action
        self['posb_action'][self._p] = posb_action        
        self['selc_action'][self._p] = selc_action
        self['reward'][self._p] = reward
        self['done'][self._p] = done
        self['next_state'][self._p] = next_state

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity
        

    def sample_discrete(self, batch_size, k=2):

        batch_size = k*self.num_action
        indices = []
        for task in range(self.num_action):
            indice = random.sample(self.action_ind[task],k)
            indices += indice
        random.shuffle(indices)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        actions = np.empty((batch_size, 1), dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        not_dones = np.empty((batch_size, 1), dtype=np.float32)
        next_states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)

        for i, index in enumerate(indices):

            states[i, ...] = self['state'][index]
            actions[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index]
            not_dones[i, ...] = self['done'][index]
            next_states[i, ...] = self['next_state'][index]

        states = torch.ByteTensor(states).to(self.device).float()/255.0 
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        not_dones = torch.FloatTensor(not_dones).to(self.device)
        next_states = torch.ByteTensor(next_states).to(self.device).float()/255.0 
        

        return states, actions, rewards, next_states, not_dones
    
    def sample_classifier(self, batch_size):

        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        posb_action = np.empty((batch_size, *(self.action_shape,)), dtype=np.float32)
        selc_action = np.empty((batch_size, *(self.action_shape,)), dtype=np.float32)

        for i, index in enumerate(indices):

            states[i, ...] = self['state'][index]
            posb_action[i, ...] = self['posb_action'][index]
            selc_action[i, ...] = self['selc_action'][index]

        states = torch.ByteTensor(states).to(self.device).float()/255.0 
        posb_action = torch.FloatTensor(posb_action).to(self.device)
        selc_action = torch.FloatTensor(selc_action).to(self.device)
        

        return states, posb_action, selc_action

    def sample_classifier2(self, batch_size):
        
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        selc_action = np.empty((batch_size, *(self.action_shape,)), dtype=np.float32)

        for i, index in enumerate(indices):
            # Convert LazeFrames into np.ndarray here.
            states[i, ...] = self['state'][index]
            selc_action[i, ...] = self['selc_action'][index]

        states = torch.ByteTensor(states).to(self.device).float()/255.0 
        selc_action = torch.FloatTensor(selc_action).to(self.device)
        

        return states, selc_action
    
    def sample_evaluate(self, batch_size, k=2):

        batch_size = k*self.num_action
        indices = []
        for task in range(self.num_action):
            indice = random.sample(self.action_ind[task],k)
            indices += indice
        random.shuffle(indices)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        actions = np.empty((batch_size, 1), dtype=np.float32)
        posb_action = np.empty((batch_size, *(self.action_shape,)), dtype=np.float32)

        for i, index in enumerate(indices):

            states[i, ...] = self['state'][index]
            actions[i, ...] = self['action'][index]
            posb_action[i, ...] = self['posb_action'][index]
            

        states = torch.ByteTensor(states).to(self.device).float()/255.0 
        actions = torch.FloatTensor(actions).to(self.device)
        posb_action = torch.FloatTensor(posb_action).to(self.device)
        

        return states, actions, posb_action

    def sample_latent(self, batch_size):

        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)

        for i, index in enumerate(indices):

            states[i, ...] = self['state'][index]

        states = torch.ByteTensor(states).to(self.device).float()/255.0 

        return states

    def __len__(self):
        return self._n
