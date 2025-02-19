import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from metadrive.envs.top_down_env import TopDownMetaDrive
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Argument parser for flexible hyperparameter loading
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="/Users/sumittade/Project/PPO(Metadrive)/MetaDrive--PPO-2/model_try1/hyperparameters.json")
args = parser.parse_args()

# Load hyperparameters from JSON file
def load_hyperparameters(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Load hyperparameters
hyperparams = load_hyperparameters(args.config)

# Extract hyperparameters
learning_rate = hyperparams["learning_rate"]
gamma = hyperparams["gamma"]
lmbda = hyperparams["lmbda"]
eps_clip = hyperparams["eps_clip"]
K_epoch = hyperparams["K_epoch"]
rollout_len = hyperparams["rollout_len"]
buffer_size = hyperparams["buffer_size"]
minibatch_size = hyperparams["minibatch_size"]
hidden_dim = hyperparams["hidden_dim"]

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to calculate CNN output size
def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride + 1

class PPO(nn.Module):
    def __init__(self, input_shape=(84, 84, 5), action_dim=2, hidden_dim=hidden_dim):
        super(PPO, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[-1], 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)  # Add BatchNorm for conv1

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)  # Add BatchNorm for conv2

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)  # Add BatchNorm for conv3

        convw = conv2d_size_out(input_shape[1], 8, 4)
        convw = conv2d_size_out(convw, 4, 2)
        convw = conv2d_size_out(convw, 3, 1)

        convh = conv2d_size_out(input_shape[0], 8, 4)
        convh = conv2d_size_out(convh, 4, 2)
        convh = conv2d_size_out(convh, 3, 1)

        linear_input_size = convw * convh * 64

        self.fc_mu = nn.Linear(linear_input_size, action_dim)
        self.fc_std = nn.Linear(linear_input_size, action_dim)
        self.fc_v = nn.Linear(linear_input_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.data = []  # ✅ Fix: Initialize data buffer here
    def put_data(self, transition):
        self.data.append(transition)  # Now self.data exists and won't cause an error
    def pi(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Now bn1 exists
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)) + 0.1
        return mu, std
        
    def v(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc_v(x)
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_batch.append(s)
            a_batch.append(a)
            r_batch.append([r])
            s_prime_batch.append(s_prime)
            prob_a_batch.append(prob_a)
            done_batch.append([0 if done else 1])
        
        return torch.tensor(np.array(s_batch), dtype=torch.float, device=device), \
               torch.tensor(np.array(a_batch), dtype=torch.float, device=device), \
               torch.tensor(np.array(r_batch), dtype=torch.float, device=device), \
               torch.tensor(np.array(s_prime_batch), dtype=torch.float, device=device), \
               torch.tensor(np.array(done_batch), dtype=torch.float, device=device), \
               torch.tensor(np.array(prob_a_batch), dtype=torch.float, device=device)
    
    def train_net(self):
        if len(self.data) < minibatch_size:
            return
        
        data = self.make_batch()
        s, a, r, s_prime, done_mask, old_log_prob = data

        with torch.no_grad():
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            advantage = delta.clone()
            for i in range(len(delta) - 2, -1, -1):
                advantage[i] += gamma * lmbda * advantage[i + 1]
        
        for _ in range(K_epoch):
            mu, std = self.pi(s)
            dist = Normal(mu, std)
            log_prob = dist.log_prob(a).sum(dim=1, keepdim=True)
            ratio = torch.exp(log_prob - old_log_prob)

            entropy = dist.entropy().sum()  # Entropy Bonus

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target) - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            
        self.data = []

if __name__ == "__main__":
    env = TopDownMetaDrive(
        dict(
            map="SSSS",
            traffic_density=0.1,
            num_scenarios=10,
            start_seed=np.random.randint(0, 1000),
        )
    )
    writer = SummaryWriter("runs/MetaDrive_PPO")
    rewards_list = []
    agent = PPO().to(device)
    obs, _ = env.reset()
    obs = np.transpose(obs, (2, 0, 1))
    
    for i in range(10000):
        mu, std = agent.pi(torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0))
        dist = Normal(mu, std)
        action = dist.sample().squeeze(0).detach().cpu().numpy()
        next_obs, reward, done, _, _ = env.step(action)
        next_obs = np.transpose(next_obs, (2, 0, 1))
        log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float, device=device)).sum().item()

        agent.put_data((obs, action, reward, next_obs, log_prob, done))
        obs = next_obs
        env.render(mode="top_down", text={"Quit": "ESC"}, film_size=(2000, 2000))
        writer.add_scalar("Reward", reward, i)

        if done:
            obs, _ = env.reset()
            obs = np.transpose(obs, (2, 0, 1))
            print(f"Episode: {i + 1}, Reward: {reward}")

        if i % 100 == 0:
            agent.train_net()
            torch.save(agent.state_dict(), "ppo_trained_model.pth")

    env.close()
    writer.close()
    print("Training complete!")
