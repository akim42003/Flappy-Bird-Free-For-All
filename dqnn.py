import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from game import FlappyBirdGame 

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output_dim)
    
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
    def forward(self, x):
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        output = self.fc5(x)
        return output

class DQLAgent:
    def __init__(self, input_dim, output_dim, device):
        self.device = device
        self.q_network = DQN(input_dim, output_dim).to(device)
        self.target_network = DQN(input_dim, output_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.buffer = ReplayBuffer(10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)  # Random action (explore)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()  # Greedy action (exploit)
    
    def train(self, batch_size):
        if len(self.buffer) < batch_size:
            return
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Q(s, a)
        q_values = self.q_network(states).gather(1, actions)
        
        # Q target
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_flappy_bird(agent, episodes, batch_size, target_update_freq):
    game = FlappyBirdGame()  # Instantiate the game
    for episode in range(episodes):
        state = game.reset()  # Reset the game and get the initial state
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)  # Choose an action
            next_state, reward, done = game.step(action)  # Perform action in the game
            agent.buffer.push(state, action, reward, next_state, done)  # Store experience
            
            state = next_state
            total_reward += reward
            
            agent.train(batch_size)  # Train the agent
        
        # Decay epsilon
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        print(f"Episode {episode}, Total Reward: {total_reward}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
input_dim = 4  # State features: [bird_y, velocity, pipe_x_distance, pipe_y_distance]
output_dim = 2  # Actions: [flap, do nothing]
agent = DQLAgent(input_dim, output_dim, device)

train_flappy_bird(agent, episodes=10000, batch_size=64, target_update_freq=10)
# Save the Q-network
torch.save(agent.q_network.state_dict(), "flappy_dql_model.pth")

