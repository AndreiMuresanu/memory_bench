from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from PIL import Image
import os

import gymnasium as gym
import gym as real_gym
import numpy as np
#import matplotlib.pyplot as plt

import wandb
from wandb.integration.sb3 import WandbCallback

from copy import deepcopy
from tqdm import tqdm
import traceback


# MONKEY PATCH START, I THINK ORDER OF IMPORTS MATTERS?

from monkey_patches import monkey_UnityParallelEnv_reset, monkey_VecEnvWrapper_get_attr, monkey_SS_concat_obs

from mlagents_envs.envs.unity_pettingzoo_base_env import UnityPettingzooBaseEnv

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.envs.unity_aec_env import UnityAECEnv
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

UnityParallelEnv.reset = monkey_UnityParallelEnv_reset

import stable_baselines3
stable_baselines3.common.vec_env.base_vec_env.VecEnvWrapper.get_attr = monkey_VecEnvWrapper_get_attr

# MONKEY PATCH END, I THINK ORDER OF IMPORTS MATTERS?

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

import supersuit as ss

from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
from sb3_contrib import RecurrentPPO



PROJECT_DIR_PATH = '/h/andrei/memory_bench'



def create_unity_env(config):
	# Setting Task Parameters
	setup_channel = EnvironmentParametersChannel()
	config_channel = EngineConfigurationChannel()

	# Setting Simulation Parameters
	config_channel.set_configuration_parameters(time_scale=100.0) # The time_scale parameter defines how quickly time will pass within the simulation
	config_channel.set_configuration_parameters(quality_level=1)  # quality_level 1 is the lowest quality, using this will improve speed

	# Setting Task Parameters
	#setup_channel.set_float_parameter("max_ingredient_count", -1)

	# select the multi-agent executable
	return UnityEnvironment(get_env_path(config),
							seed=config['seed'],
							side_channels=[setup_channel, config_channel])


def make_env(config):
	try: unity_env.close()
	except: pass
	try: env.close()
	except: pass
	
	unity_env = create_unity_env(config)
	
	if config['parallelism'] == 'single_agent':
		return UnityToGymWrapper(unity_env, uint8_visual=True)
	
	elif config['parallelism'] == 'multi_agent':
		
		env = UnityParallelEnv(unity_env)
		
		for key, value in env._observation_spaces.items():
			env._observation_spaces[key] = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
		
		env = ss.pettingzoo_env_to_vec_env_v1(env)
		env.black_death = True	# THIS IS WRONG. We are just suppressing the error. Look here: https://github.com/Farama-Foundation/SuperSuit/blob/master/supersuit/vector/markov_vector_wrapper.py#L38

		env = VecMonitor(venv=env)
		env.venv.concat_obs = monkey_SS_concat_obs.__get__(env.venv, ss.vector.markov_vector_wrapper.MarkovVectorEnv)
		
		return env

	else:
		raise ValueError()


class MyCallback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""

	def __init__(self, verbose=2):
		super().__init__(verbose)
		self.rollout_partial_cum_batch_reward = np.zeros(24)

	def _on_rollout_end(self) -> None:
		avg_cum_batch_reward = 0
		for cum_reward in self.rollout_partial_cum_batch_reward:
			avg_cum_batch_reward += cum_reward
		avg_cum_batch_reward /= 24
		self.logger.record("rollout/avg_cum_batch_reward", avg_cum_batch_reward)
		self.rollout_partial_cum_batch_reward = np.zeros(24)	# reset for the next rollout
	
	def _on_step(self) -> bool:
		self.rollout_partial_cum_batch_reward += self.locals['rewards']
				
		avg_cum_batch_reward = 0
		for cum_reward in self.rollout_partial_cum_batch_reward:
			avg_cum_batch_reward += cum_reward
		avg_cum_batch_reward /= 24
		self.logger.record("train/avg_cum_batch_reward", avg_cum_batch_reward)

		return True


def sb_training(config):
	run = wandb.init(
		project="ReMEMber",
		config=config,
		sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
		monitor_gym=True,  # auto-upload the videos of agents playing the game
		save_code=True,  # optional
		mode='disabled'	# this makes it so nothing is logged and useful to avoid logging debugging runs
	)

	env = make_env(config)

	algo = get_algo(config['algo_name'])
	model = algo(config["policy_type"], env, verbose=config['verbosity'], tensorboard_log=f"runs/{run.id}")
	model.learn(
		total_timesteps=config["total_timesteps"],
		#callback=WandbCallback(
		#	gradient_save_freq=100,
		#	model_save_path=f"models/{run.id}",
		#	verbose=config['verbosity'],
		#),
		callback=MyCallback(),
		progress_bar=True
	)

	#print("Saving model to sb_model.pkl")
	#model.save("./models/sb_test_model.pkl")

	env.close()
	run.finish()


def get_env_path(config):
	if config['parallelism'] == 'multi_agent':
		parallelism_folder = 'multi_agent'
	elif config['parallelism'] == 'single_agent':
		parallelism_folder = 'single_agent'
	else:
		raise ValueError()

	return f'{PROJECT_DIR_PATH}/Builds/{config["env_name"]}/standard/linux/pixel_input/{parallelism_folder}/gamefile.x86_64'


def get_algo(algo_name):
	algos = {
		'PPO': PPO,
		'RecurrentPPO': RecurrentPPO,
		'SAC': SAC,
		'A2C': A2C,
		'DQN': DQN,
	}
	return algos[algo_name]


if __name__ == '__main__':

	task_names = [
		'AllergicRobot',
		#'MatchingPairs',
		#'Hallway',
		#'RecipeRecall'
	]

	# can't store the algos directly because we want to be able to directly upload the config dict to wandb
	algo_names = [
		'PPO',
		#'RecurrentPPO',
		#'A2C',
		#'DQN',
		#'SAC',
	]

	#num_of_trial_repeats = 5
	num_of_trial_repeats = 1
	
	base_config = {
		"policy_type": "CnnPolicy",
		#"total_timesteps": 250_000,
		#"total_timesteps": 3,
		"total_timesteps": 100_000,
		#"parallelism": "single_agent",
		"parallelism": "multi_agent",
		#"verbosity": 0,
		"verbosity": 2,
	}

	for task_name in tqdm(task_names, desc='tasks completed'):
		for algo_name in tqdm(algo_names, desc='algos completed'):
			for trail_num in tqdm(range(num_of_trial_repeats), desc='trails completed'):
				config = deepcopy(base_config)	# I think deepcopy is likely not needed
				config['env_name'] = task_name
				config['algo_name'] = algo_name
				config['trail_num'] = trail_num
				config['seed'] = trail_num

				# convert all this sloppy code into a factory
				if algo_name == 'RecurrentPPO':
					config['policy_type'] = 'CnnLstmPolicy'
				
				try:
					sb_training(config)
					#multi_agent_input_checking(config)
				except:
					print('\n\n||||||||||||||||||||||||| ERROR |||||||||||||||||||||||||\n')
					traceback.print_exc()
					print('\n||||||||||||||||||||||||| ERROR |||||||||||||||||||||||||\n\n')

	print('\nDONE')
