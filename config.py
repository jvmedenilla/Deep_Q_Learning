# Hyperparameters for DQN agent, memory and training
EPISODES = 3
HEIGHT = 84
WIDTH = 84
HISTORY_SIZE = 6
learning_rate = 0.001
lstm_seq_length = 20
evaluation_reward_length = 100
Memory_capacity = 1000000
train_frame = 100 #100000
batch_size = 64
scheduler_gamma = 0.4
scheduler_step_size = 100000

# Hyperparameters for Double DQN agent
update_target_network_frequency = 1000