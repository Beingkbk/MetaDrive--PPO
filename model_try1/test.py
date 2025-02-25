import torch
import numpy as np
from metadrive.envs.top_down_env import TopDownMetaDrive
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Load the trained model
def load_model(model_path, input_shape=(84, 84, 5), action_dim=2, hidden_dim=64):
    class PPO(nn.Module):
        def __init__(self, input_shape, action_dim, hidden_dim):
            super(PPO, self).__init__()
            self.conv1 = nn.Conv2d(input_shape[-1], 32, kernel_size=8, stride=4)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(64)

            # Calculate the output size of the CNN layers
            def conv2d_size_out(size, kernel_size, stride):
                return (size - (kernel_size - 1) - 1) // stride + 1

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

        def pi(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = x.view(x.size(0), -1)
            mu = 2.0 * torch.tanh(self.fc_mu(x))
            std = F.softplus(self.fc_std(x)) + 0.1
            return mu, std

    model = PPO(input_shape, action_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Test the trained model
def test_model(model_path, num_episodes=10):
    # Initialize the environment
    env = TopDownMetaDrive(
        dict(
            map="SSSS",
            traffic_density=0.1,
            num_scenarios=10,
            start_seed=np.random.randint(0, 1000),
        )
    )

    # Load the trained model
    model = load_model(model_path)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = np.transpose(obs, (2, 0, 1))  # Change shape to (C, H, W)
        done = False
        total_reward = 0

        while not done:
            # Get the action from the model
            with torch.no_grad():
                mu, std = model.pi(torch.tensor(obs, dtype=torch.float).unsqueeze(0))
                dist = Normal(mu, std)
                action = dist.sample().squeeze(0).detach().numpy()

            # Take the action in the environment
            next_obs, reward, done, _, _ = env.step(action)
            next_obs = np.transpose(next_obs, (2, 0, 1))  # Change shape to (C, H, W)
            total_reward += reward

            # Render the environment (optional)
            env.render(mode="top_down", text={"Quit": "ESC"}, film_size=(2000, 2000))

            # Update the observation
            obs = next_obs

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    env.close()
    print("Testing complete!")

# Run the test
if __name__ == "__main__":
    model_path = "ppo_trained_model.pth"  # Path to your trained model
    test_model(model_path, num_episodes=10)  # Test for 10 episodes
