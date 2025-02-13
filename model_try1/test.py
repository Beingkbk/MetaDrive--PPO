import torch
from metadrive.envs.top_down_env import TopDownMetaDrive
from policy import PPO  # Ensure policy.py contains your PPO implementation
import random

def test_trained_model():
    env = TopDownMetaDrive(
        dict(
            map="SSSS",
            traffic_density=0.1,
            num_scenarios=100,
            start_seed=random.randint(0, 1000),
        )
    )
    
    # Initialize and load trained model
    agent = PPO(input_dim=35280, action_dim=2)
    agent.load_state_dict(torch.load("D:\PPO\PPO_project_extended\MetaDrive-PPO-Agent\ppo_trained_model.pth"))
    agent.eval()
    
    try:
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(1000):  # Run for a fixed number of steps
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
                mu, _ = agent.pi(obs_tensor)
                action = mu.squeeze().numpy()
            
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            env.render(mode="top_down", text={"Total Reward": total_reward})
            
            if done or truncated:
                break
        
        print(f"Total reward after testing: {total_reward}")
    finally:
        env.close()

if __name__ == "__main__":
    test_trained_model()
