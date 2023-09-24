import gymnasium as gym

#from stable_baselines3 import deepq
from stable_baselines3 import PPO
#from stable_baselines3 import logger

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper


def main():
	path_to_env = r'C:\Users\Andrei\Documents\vector_institute\memory_palace_stuff\unity_projects\memory_palace_2\Builds\windows'
	unity_env = UnityEnvironment(path_to_env)
	env = UnityToGymWrapper(unity_env, uint8_visual=True)
	#logger.configure('./logs')  # Change to log in a different directory
	act = PPO.learn(
		env,
		"cnn",  # For visual inputs
		lr=2.5e-4,
		total_timesteps=1000000,
		buffer_size=50000,
		exploration_fraction=0.05,
		exploration_final_eps=0.1,
		print_freq=20,
		train_freq=5,
		learning_starts=20000,
		target_network_update_freq=50,
		gamma=0.99,
		prioritized_replay=False,
		checkpoint_freq=1000,
		checkpoint_path='./logs',  # Change to save model in a different directory
		dueling=True
	)
	print("Saving model to unity_model.pkl")
	act.save("unity_model.pkl")


if __name__ == '__main__':
	main()