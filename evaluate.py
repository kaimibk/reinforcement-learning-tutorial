from stable_baselines3 import PPO
from custom_env import ServerRoomEnv

# Load the trained model
model = PPO.load("./server_room_ppo_model")
env = ServerRoomEnv(verbose=True)  # Set verbose to True to see detailed output

# Enjoy a 5-episode test drive
for ep in range(5):
    obs, info = env.reset()
    done = False
    score = 0
    
    print(f"\n--- Episode {ep+1} ---")
    while not done:
        # Instead of random.sample(), we ask our trained model for the best action!
        # _states is used for recurrent policies (LSTMs), which we aren't using here.
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        env.render()
        
        done = terminated or truncated

    print(f"Total Reward: {score:.2f}")

env.close()
