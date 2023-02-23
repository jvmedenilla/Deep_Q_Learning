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
        self.target_net.to(device)
        #self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        
        # Initialize a target network and initialize the target network to the policy net
        self.update_target_net()

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)
        
    def update_target_net(self):
         self.target_net.load_state_dict(self.policy_net.state_dict())

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        
        #print(state)
        if np.random.rand() <= self.epsilon:
            
            #print("here")
            a = torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
            return a
            
        else:
            #print("ye")
            #print(self.policy_net(state).parameters())
            with torch.no_grad():
                  state = torch.FloatTensor(state).unsqueeze(0).cuda()  
            return self.policy_net(state).max(1)[1].view(1, 1)


    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()
        #print(np.dtype(mini_batch))
        #mini_batch = torch.tensor(mini_batch)

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        next_states = torch.tensor(next_states).cuda()
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8)


        # Compute Q(s_t, a), the Q-value of the current state
        state_action_values = self.policy_net(states).gather(1, actions.view(batch_size, -1))

        # Compute Q function of next state and
        # Find maximum Q-value of action at next state from policy net
        next_state_values = torch.zeros(batch_size,device=device).cuda()
        non_final_mask=torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([i for i in next_states if i is not None]).view(states.size()).cuda()
        # Compute the expected Q values
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

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