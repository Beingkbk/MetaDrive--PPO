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
parser.add_argument("--config", type=str, default="model_try1/hyperparameters.json")
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
        self.data = []  # Fix: Initialize data buffer here
    def put_data(self, transition):
        self.data.append(transition)  # Now self.data exists and won't cause an error
    def pi(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Now bn1 exists
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        self.dropout = nn.Dropout(p=0.2)

        
        x = x.view(x.size(0), -1)
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)) + 0.5
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
            #use_render=True,  # render the environment
            map="S",  # Simple straight road
            traffic_density=0.0,  # No traffic
            num_scenarios=10,
            start_seed=np.random.randint(0, 1000),
        )
    )
    writer = SummaryWriter("runs/MetaDrive_PPO")
    rewards_list = []
    agent = PPO().to(device)
    
    timeout_steps = 100000 # Timeout after 10,000 steps (to prevent infinite loops)

    for i in range(10000):  # Increased training episodes
        episode_reward = 0  # Track total reward for the episode
        episode_steps = 0  # Track number of steps in the episode
        done = False

        # Reset the environment
        obs, _ = env.reset()
        obs = np.transpose(obs, (2, 0, 1))
        episode_reward = 0
        done = False

        # Initialize prev_distance before entering the loop
        prev_distance = np.linalg.norm(env.agent.position - env.agent.navigation.final_lane.end)

        while not done:
            # Check if the episode has timed out
            if episode_steps >= timeout_steps:
                print("Episode timed out. Resetting environment.")
                done = True
                break

            # Get the action from the agent
            with torch.no_grad():
                mu, std = agent.pi(torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0))
                dist = Normal(mu, std)
                action = dist.sample().squeeze(0).detach().cpu().numpy()

                # Take the action in the environment
                next_obs, reward, done, _, info = env.step(action)

                # Adjust rewards
                if done and info["arrive_dest"]:
                    reward += 100  # Large reward for reaching the destination

                # Check if the agent is off the road using env.agent
                if not env.agent.on_lane:  # Use the agent's on_lane property
                    reward -= 5  # Penalty for going off the road
                    print("Agent is off the road! Penalty applied.")  # Debug print statement
                    done = True  # Manually set done to True

                # Add a reward for maintaining a higher speed
                agent_speed = env.agent.speed  # Speed in m/s
                speed_reward = 0.1 * agent_speed  # Reward proportional to speed
                reward += speed_reward

                # Add a reward for staying on the road
                if env.agent.on_lane:
                    reward += 0.1  # Small reward for staying on the road

                # Add a reward for making progress toward the destination
                distance_to_dest = np.linalg.norm(env.agent.position - env.agent.navigation.final_lane.end)
                progress_reward = 0.1 * (prev_distance - distance_to_dest)# Reward for reducing distance to destination
                reward += progress_reward
                prev_distance = distance_to_dest


                # Log agent's actions and state
                print(f"Episode: {i + 1}, Steps: {episode_steps}, Total Reward: {episode_reward}")
                print(f"Agent Position: {env.agent.position}, Speed: {env.agent.speed}, Distance to Destination: {distance_to_dest}")
                print(f"Steering: {action[0]}, Throttle: {action[1]}")

                # Update episode reward and steps
                episode_reward += reward
                episode_steps += 1

                # Store the transition in the agent's memory
                next_obs = np.transpose(next_obs, (2, 0, 1))  # Change shape to (C, H, W)
                log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float, device=device)).sum().item()
                agent.put_data((obs, action, reward, next_obs, log_prob, done))

                # Render the environment (optional)
                #env.render(mode="top_down", text={"Quit": "ESC"}, film_size=(2000, 2000))

                # Update the observation
                obs = next_obs

        # Log episode details
        print(f"Episode: {i + 1}, Steps: {episode_steps}, Total Reward: {episode_reward}")
        writer.add_scalar("Episode Reward", episode_reward, i)
        writer.add_scalar("Episode Steps", episode_steps, i)

        # Train the agent after each episode
        agent.train_net()


        Exp_no = 1 # Experiment number
        # Save the model periodically
        if i % 100 == 0:
            torch.save(agent.state_dict(), f"ppo_trained_model_{Exp_no}.pth")

    env.close()
    writer.close()
    print("Training complete!")