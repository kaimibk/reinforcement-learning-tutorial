from stable_baselines3 import PPO

from custom_env import ServerRoomEnv  # Import your custom environment

# 1. Instantiate the Environment
# Optional: Pass hyperparameters to the environment constructor if needed (e.g., target_temp=22)
env = ServerRoomEnv()

# 2. Instantiate the PPO Agent
# NOTE: Because our observation space is a Dict, we MUST use "MultiInputPolicy"
# If it was just a Box, we would use "MlpPolicy"
model = PPO("MultiInputPolicy", env, verbose=1)

print("Starting training...")
# 3. Train the Agent! (10,000 steps is very fast, usually takes < 1 minute)
model.learn(total_timesteps=10_000)
print("Training finished!")

# 4. Save the "Brain"
model.save("./server_room_ppo_model")
