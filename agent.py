import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory, ReplayMemoryLSTM
from model import DQN, DQN_LSTM
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        self.target_net = DQN(action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        
        #print(state)
        if np.random.rand() <= self.epsilon:
            #a = torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
            a = random.randrange(self.action_size)
            print("here",type(a))
            #print(a)
            
        else:
            #print(self.policy_net(state).parameters())
            a = self.policy_net(state).max(1)[1].view(1, 1)
            print("ye", type(a))
            
        return a

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8)

        # Compute Q(s_t, a), the Q-value of the current 
        #print(type(states))
        state_action_values = self.policy_net(states).gather(1, actions)

        # Compute Q function of next state and
        # Find maximum Q-value of action at next state from policy net
        next_state_values = torch.zeros(mini_batch, device=device)
        next_state_values[mask] = self.target_net(next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.discount_factor) + rewards

        # Compute the Huber Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model, .step() both the optimizer and the scheduler!
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    

class LSTM_Agent(Agent):
    def __init__(self, action_size):
        super().__init__(action_size)
        # Generate the memory
        self.memory = ReplayMemoryLSTM()

        # Create the policy net
        self.policy_net = DQN_LSTM(action_size)
        self.policy_net.to(device)
        
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state, hidden=None):
        
        state = torch.FloatTensor(state).unsqueeze(0).cuda() 
        a, hidden = self.policy_net(state)
        
        if np.random.rand() <= self.epsilon:
            
            a = torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

        else:
            #print(self.policy_net(state).parameters())
            #state = torch.FloatTensor(state).unsqueeze(0).cuda()  
            a = self.policy_net(state)[0].max(1)[1].view(1, 1)
      
        
        return a, hidden

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :lstm_seq_length, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        next_states = torch.tensor(next_states).cuda()
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8)

        ### All the following code is nearly same as that for Agent

        #print(states[0].size()[0])
        # Compute Q(s_t, a), the Q-value of the current state
        #for i in range(states[0].size()[0]):
        
        a,b,c,d = states.size()
        ### All the following code is nearly same as that for Agent

        #print(states[0].size()[0])
        # Compute Q(s_t, a), the Q-value of the current state
        #for i in range(states[0].size()[0]):
        states = torch.reshape(states, (a*b, 1, c,d))
        #print(states.size())
        state_action_values = self.policy_net(states)[0].gather(1, actions.view(batch_size, -1))
        #states = torch.reshape(states, (a,b,c,d))
        # Compute Q function of next state and
        # Find maximum Q-value of action at next state from policy net
        next_state_values = torch.zeros(batch_size,device=device).cuda()
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([i for i in next_states if i is not None]).view(states.size()).cuda()
        # Compute the expected Q values
        #print(non_final_mask.size(), non_final_next_states.size())
        #non_final_mask = torch.reshape(non_final_mask, (a*b))
        #non_final_next_states = non_final_next_states.view(states)
        #non_final_next_states = torch.reshape(non_final_next_states, (a*b, 1, c,d))
        #next_state_values[non_final_mask] = self.policy_net(non_final_next_states)[0].max(1)[0].detach()
        non_final_next_states_last = self.policy_net(non_final_next_states)[0]
        non_final_next_states_last = torch.reshape(non_final_next_states_last, (a,b,3))
        #print(next_state_values.size(),non_final_mask.size(),non_final_next_states_last.size(), non_final_next_states_last[:,-1:,:].view(-1,3).size())
        #print(non_final_next_states_last[:,-1:,:].max(1))
        next_state_values[non_final_mask] = non_final_next_states_last[:,-1:,:].view(-1,3).max(1)[0].detach()
    
        expected_state_action_values = (next_state_values[0] * self.discount_factor) + rewards

        # Compute the Huber Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model, .step() both the optimizer and the scheduler!
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


