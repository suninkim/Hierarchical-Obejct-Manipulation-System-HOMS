from collections import deque
import numpy as np
import torch
import time

class Memory(dict):
    ''' Memory is memory-efficient but time-inefficient. '''
    keys = ['state', 'robot_state', 'action', 'reward', 'done', 'next_state', 'next_robot_state', 'task']

    def __init__(self, capacity, observation_shape, robot_shape, action_shape, task_shape, reward_ratio, task_ratio, num_step_after_done, batch_size, device):
        super(Memory, self).__init__()
        self.capacity = int(1)
        self.observation_shape = observation_shape
        self.robot_shape = robot_shape
        self.action_shape = action_shape
        self.task_shape = task_shape
        self.batch_size = batch_size
        self.device = device

        self.reward_ind = []        
        self.task_index = []
        self.aver_reward_step = []

        self.reward_ratio = reward_ratio
        self.task_ratio = task_ratio
        self.num_step_after_done = num_step_after_done
        self.reset()

    def reset(self):
        self.is_set_init = False
        self._p = 0
        self._n = 0
        for key in self.keys:
            self[key] = [None] * self.capacity
        print("RESET")

    def cal_sampling_ratio(self):
        total_step = np.array(self.aver_reward_step).sum()
        balanced_sample = np.array(self.aver_reward_step)/total_step
        self.balanced_sample = np.round(balanced_sample*self.batch_size).astype(np.uint8)
        print(f'Data for each task = {self.balanced_sample}')

        self.reward_ind = np.array(self.reward_ind)
        self.non_reward_ind = np.delete(np.arange(self._n),self.reward_ind)

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

    def add_data_to_buffer(self, data_path_list):
        data_list, buffer_size = self.get_data(data_path_list)
        self.capacity += buffer_size

        add_start_index = self._p
        
        aver_reward_ind = 0
        skip = 0

        total_epi  = 0
        epi_success = 0
        for data in data_list:
            # print(data)
            
            for j in range(len(data)):
                assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
                    data[j]['next_observations']))

                bf_r = 0
                done_ind = 100
                total_epi += 1
                for i in range(len(data[j]['actions'])):

                    if i > done_ind+self.num_step_after_done:
                        skip += 1
                        break

                    task = np.zeros((self.task_shape[0],))
                    task[data[j]['tasks'][i]] = 1

                    state = data[j]['observations'][i]['image'][:3,:,:]
                    robot_state = data[j]['observations'][i]['robot']

                    action = data[j]['actions'][i]  

                    not_done = [not(data[j]['terminals'][i])]
                    reward = data[j]['rewards'][i]

                    if reward != bf_r:
                        self.reward_ind.append(self._p)
                        aver_reward_ind += i
                        done_ind = i
                        epi_success += 1

                    bf_r = reward

                    if self._n < self.capacity:
                        for key in self.keys:
                            self[key] += [None]                       

                    next_state = data[j]['next_observations'][i]['image'][:3,:,:]
                    next_robot_state = data[j]['next_observations'][i]['robot']


                    self.append(state, robot_state, action, reward, not_done, next_state, next_robot_state, task)

        
        aver_reward_ind = aver_reward_ind/(len(data_list)*len(data_list[0]))
        epi_length = len(data_list[0][0]['actions'])
        
        add_end_index = self._p -1
        self.task_index.append([add_start_index, add_end_index])
        self.aver_reward_step.append(aver_reward_ind)

        print(f"capacity : {self.capacity}")
        print(f"p : {self._p}")
        print(f"n : {self._n}")
        print(f'skiped episode: {skip}')
        print(f'average reward index = {aver_reward_ind}, episode length = {epi_length}')
        print(f'success rate = {100*epi_success/total_epi}')
        data_list=[]

    def append(self, state, robot_state, action, reward, done, next_state, next_robot_state, task):
        self['state'][self._p] = state
        self['robot_state'][self._p] = robot_state
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done
        self['next_state'][self._p] = next_state
        self['next_robot_state'][self._p] = next_robot_state
        self['task'][self._p] = task

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity
        # print(f"Cur position : {self._p}")

    def sample_latent(self, batch_size):

        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)

        for i, index in enumerate(indices):
            states[i, ...] = self['state'][index]
            
        states = torch.ByteTensor(states).to(self.device).float()/255.0 
        
        return states

    def sample_sac(self, batch_size):

        if self.task_ratio:
            batch_size = np.sum(self.balanced_sample)
            indices = np.array([],dtype=np.uint8)
            for task in range(self.task_shape[0]-1):
                low, high = self.task_index[task][0], self.task_index[task][1]
                indice = np.random.choice(np.arange(low,high+1), self.balanced_sample[task], replace=False)
                indices = np.concatenate((indices,indice),axis=0)
        else:
            indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        robot_states = np.empty((batch_size, *self.robot_shape), dtype=np.float32)
        actions = np.empty((batch_size, *self.action_shape), dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        not_dones = np.empty((batch_size, 1), dtype=np.float32)
        next_states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        next_robot_states = np.empty((batch_size, *self.robot_shape), dtype=np.float32)
        tasks = np.empty((batch_size, *self.task_shape), dtype=np.float32)

        for i, index in enumerate(indices):
            states[i, ...] = self['state'][index]            
            robot_states[i, ...] = self['robot_state'][index]
            actions[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index]
            not_dones[i, ...] = self['done'][index]
            next_states[i, ...] = self['next_state'][index]
            next_robot_states[i, ...] = self['next_robot_state'][index]
            tasks[i, ...] = self['task'][index]

        states = torch.ByteTensor(states).to(self.device).float()/255.0 
        robot_states = torch.FloatTensor(robot_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        not_dones = torch.FloatTensor(not_dones).to(self.device)
        next_states = torch.ByteTensor(next_states).to(self.device).float()/255.0 
        next_robot_states = torch.FloatTensor(next_robot_states).to(self.device)
        tasks = torch.FloatTensor(tasks).to(self.device)

        return states, robot_states,  actions, rewards, next_states, next_robot_states, not_dones, tasks

    def sample_reward(self, batch_size):

        indices = np.random.choice(self.reward_ind, batch_size, replace=False)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        robot_states = np.empty((batch_size, *self.robot_shape), dtype=np.float32)
        actions = np.empty((batch_size, *self.action_shape), dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        not_dones = np.empty((batch_size, 1), dtype=np.float32)
        next_states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        next_robot_states = np.empty((batch_size, *self.robot_shape), dtype=np.float32)
        tasks = np.empty((batch_size, *self.task_shape), dtype=np.float32)

        for i, index in enumerate(indices):
            states[i, ...] = self['state'][index]            
            robot_states[i, ...] = self['robot_state'][index]
            actions[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index]
            not_dones[i, ...] = self['done'][index]
            next_states[i, ...] = self['next_state'][index]
            next_robot_states[i, ...] = self['next_robot_state'][index]
            tasks[i, ...] = self['task'][index]

        states = torch.ByteTensor(states).to(self.device).float()/255.0 
        robot_states = torch.FloatTensor(robot_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        not_dones = torch.FloatTensor(not_dones).to(self.device)
        next_states = torch.ByteTensor(next_states).to(self.device).float()/255.0 
        next_robot_states = torch.FloatTensor(next_robot_states).to(self.device)
        tasks = torch.FloatTensor(tasks).to(self.device)

        return states, robot_states,  actions, rewards, next_states, next_robot_states, not_dones, tasks
    
    def sample_non_reward(self, batch_size):

        if self.task_ratio:
            batch_size = np.sum(self.balanced_sample)
            indices = np.array([],dtype=np.uint8)
            for task in range(self.task_shape[0]-1):
                low, high = self.task_index[task][0], self.task_index[task][1]
                task_reward_ind = self.non_reward_ind[(self.non_reward_ind>=low) & (self.non_reward_ind<=high)]
                indice = np.random.choice(task_reward_ind, self.balanced_sample[task], replace=False)
                indices = np.concatenate((indices,indice),axis=0)
        else:
            indices = np.random.choice(self.non_reward_ind, batch_size, replace=False)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        robot_states = np.empty((batch_size, *self.robot_shape), dtype=np.float32)
        actions = np.empty((batch_size, *self.action_shape), dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        not_dones = np.empty((batch_size, 1), dtype=np.float32)
        next_states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        next_robot_states = np.empty((batch_size, *self.robot_shape), dtype=np.float32)
        tasks = np.empty((batch_size, *self.task_shape), dtype=np.float32)

        for i, index in enumerate(indices):
            states[i, ...] = self['state'][index]            
            robot_states[i, ...] = self['robot_state'][index]
            actions[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index]
            not_dones[i, ...] = self['done'][index]
            next_states[i, ...] = self['next_state'][index]
            next_robot_states[i, ...] = self['next_robot_state'][index]
            tasks[i, ...] = self['task'][index]

        states = torch.ByteTensor(states).to(self.device).float()/255.0 
        robot_states = torch.FloatTensor(robot_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        not_dones = torch.FloatTensor(not_dones).to(self.device)
        next_states = torch.ByteTensor(next_states).to(self.device).float()/255.0 
        next_robot_states = torch.FloatTensor(next_robot_states).to(self.device)
        tasks = torch.FloatTensor(tasks).to(self.device)

        return states, robot_states,  actions, rewards, next_states, next_robot_states, not_dones, tasks

    def sample_reward_ratio(self, batch_size):        

        if self.task_ratio:
            batch_size = np.sum(self.balanced_sample)
            indices = np.array([],dtype=np.uint8)
            for task in range(self.task_shape[0]-1):
                low, high = self.task_index[task][0], self.task_index[task][1]
                task_reward_ind = self.reward_ind[(self.reward_ind>=low) & (self.reward_ind<=high)]
                task_non_reward_ind = self.non_reward_ind[(self.non_reward_ind>=low) & (self.non_reward_ind<=high)]
                reward_num = np.round(self.balanced_sample[task]/(self.reward_ratio+1)).astype(np.uint8)                
                non_reward_num = self.balanced_sample[task] - reward_num
                reward_indice = np.random.choice(task_reward_ind, reward_num, replace=False)
                non_reward_indice = np.random.choice(task_non_reward_ind, non_reward_num, replace=False)
                indices = np.concatenate((indices,reward_indice,non_reward_indice),axis=0)
        else:
            indices = np.array([],dtype=np.uint8)
            reward_num = np.round(batch_size/(self.reward_ratio+1)).astype(np.uint8)                
            non_reward_num = batch_size - reward_num
            reward_indice = np.random.choice(self.reward_ind, reward_num, replace=False)
            non_reward_indice = np.random.choice(self.non_reward_ind, non_reward_num, replace=False)
            indices = np.concatenate((indices,reward_indice,non_reward_indice),axis=0)
            indices = np.random.choice(self.non_reward_ind, batch_size, replace=False)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        robot_states = np.empty((batch_size, *self.robot_shape), dtype=np.float32)
        actions = np.empty((batch_size, *self.action_shape), dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        not_dones = np.empty((batch_size, 1), dtype=np.float32)
        next_states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        next_robot_states = np.empty((batch_size, *self.robot_shape), dtype=np.float32)
        tasks = np.empty((batch_size, *self.task_shape), dtype=np.float32)

        for i, index in enumerate(indices):
            states[i, ...] = self['state'][index]            
            robot_states[i, ...] = self['robot_state'][index]
            actions[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index]
            not_dones[i, ...] = self['done'][index]
            next_states[i, ...] = self['next_state'][index]
            next_robot_states[i, ...] = self['next_robot_state'][index]
            tasks[i, ...] = self['task'][index]

        states = torch.ByteTensor(states).to(self.device).float()/255.0 
        robot_states = torch.FloatTensor(robot_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        not_dones = torch.FloatTensor(not_dones).to(self.device)
        next_states = torch.ByteTensor(next_states).to(self.device).float()/255.0 
        next_robot_states = torch.FloatTensor(next_robot_states).to(self.device)
        tasks = torch.FloatTensor(tasks).to(self.device)

        return states, robot_states,  actions, rewards, next_states, next_robot_states, not_dones, tasks
    
    def sample_task(self, batch_size, task):
        
        low, high = self.task_index[task][0], self.task_index[task][1]
        indices = np.random.randint(low=low, high=high, size=batch_size)
        
        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        robot_states = np.empty((batch_size, *self.robot_shape), dtype=np.float32)
        actions = np.empty((batch_size, *self.action_shape), dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        not_dones = np.empty((batch_size, 1), dtype=np.float32)
        next_states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        next_robot_states = np.empty((batch_size, *self.robot_shape), dtype=np.float32)
        tasks = np.empty((batch_size, *self.task_shape), dtype=np.float32)

        for i, index in enumerate(indices):
            states[i, ...] = self['state'][index]            
            robot_states[i, ...] = self['robot_state'][index]
            actions[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index]
            not_dones[i, ...] = self['done'][index]
            next_states[i, ...] = self['next_state'][index]
            next_robot_states[i, ...] = self['next_robot_state'][index]
            tasks[i, ...] = self['task'][index]

        states = torch.ByteTensor(states).to(self.device).float()/255.0 
        robot_states = torch.FloatTensor(robot_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        not_dones = torch.FloatTensor(not_dones).to(self.device)
        next_states = torch.ByteTensor(next_states).to(self.device).float()/255.0 
        next_robot_states = torch.FloatTensor(next_robot_states).to(self.device)
        tasks = torch.FloatTensor(tasks).to(self.device)

        return states, robot_states,  actions, rewards, next_states, next_robot_states, not_dones, tasks
    
    def __len__(self):
        return self._n
