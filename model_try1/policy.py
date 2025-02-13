import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from metadrive.envs.top_down_env import TopDownMetaDrive
import numpy as np
import json 

# Load hyperparameters from JSON file
def load_hyperparameters(filepath="D:\PPO\PPO_project_extended\MetaDrive-PPO-Agent\ppo_model_try8\hyperparameters.json"):
    with open(filepath, "r") as f:
        params = json.load(f)
    return params

# Load the hyperparameters
hyperparams = load_hyperparameters()

# Use them in your PPO class
learning_rate = hyperparams["learning_rate"]
gamma = hyperparams["gamma"]
lmbda = hyperparams["lmbda"]
eps_clip = hyperparams["eps_clip"]
K_epoch = hyperparams["K_epoch"]
rollout_len = hyperparams["rollout_len"]
buffer_size = hyperparams["buffer_size"]
minibatch_size = hyperparams["minibatch_size"]
hidden_dim = hyperparams["hidden_dim"]

class PPO(nn.Module):
    def __init__(self, input_dim=35280, action_dim=2, hidden_dim=hidden_dim):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))  # Steering & Throttle range [-2,2]
        std = F.softplus(self.fc_std(x)) + 0.1  # Ensure std is positive
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
            done_mask = 0 if done else 1
            done_batch.append([done_mask])
        
        return torch.tensor(np.array(s_batch), dtype=torch.float), \
                torch.tensor(np.array(a_batch), dtype=torch.float), \
                torch.tensor(np.array(r_batch), dtype=torch.float), \
                torch.tensor(np.array(s_prime_batch), dtype=torch.float), \
                torch.tensor(np.array(done_batch), dtype=torch.float), \
                torch.tensor(np.array(prob_a_batch), dtype=torch.float)
    
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

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target)
            
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
            num_scenarios=100,
            start_seed=np.random.randint(0, 1000),
        )
    )
    agent = PPO()
    obs, _ = env.reset()
    obs = obs.flatten()  # Flatten observation
    print("Observation shape:", obs.shape)

    for i in range(10000):
        mu, std = agent.pi(torch.tensor(obs, dtype=torch.float).unsqueeze(0))
        dist = Normal(mu, std)
        action = dist.sample().squeeze(0).detach().numpy()
        next_obs, reward, done, _, _ = env.step(action)
        next_obs = next_obs.flatten()
        log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float)).sum().item()

        agent.put_data((obs, action, reward, next_obs, log_prob, done))
        obs = next_obs

        env.render(mode="top_down", text={"Quit": "ESC"}, film_size=(2000, 2000))
        
        if done:
            obs, info = env.reset()
            obs = obs.flatten()
            print(f"Episode: {i + 1}, Reward: {reward}, Done: {done}, Info: {info}") #! Added info, while printing info it is showing "policy" as "EnvInputpolicy", but when ran it in debug mode it is using above PPO class for training.
        if i % 100 == 0:
            agent.train_net()
            # Save trained model
            torch.save(agent.state_dict(), "ppo_trained_model.pth")
            print("Model saved successfully.")
    env.close()
    print("Training finished.")