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
import sys
import json

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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

import supersuit as ss

from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
from sb3_contrib import RecurrentPPO
import statistics

from nasty_evaluate_policy import nasty_evaluate_policy


PROJECT_DIR_PATH = '/h/andrei/memory_bench'



# TODO: make this function take in actual params not just the config
def create_unity_env(config, worker_id=0):
	# Setting Task Parameters
	setup_channel = EnvironmentParametersChannel()
	config_channel = EngineConfigurationChannel()

	# Setting Simulation Parameters
	config_channel.set_configuration_parameters(time_scale=100.0) # The time_scale parameter defines how quickly time will pass within the simulation
	config_channel.set_configuration_parameters(quality_level=1)  # quality_level 1 is the lowest quality, using this will improve speed

	# Setting Task Parameters
	if 'task_configs' in config:
		for config_name, config_value in config['task_configs'].items():
			setup_channel.set_float_parameter(config_name, config_value)

	if config['env_name'] != 'Hallway':
		setup_channel.set_float_parameter('initialization_seed', config['seed'])	# the initialization_seed controls environment initialization (ex., food tastiness in AllegicRobot). Not all envs have this parameter

	# Select the multi-agent executable
	kwargs = {
		'seed': config['seed'],
		'side_channels': [setup_channel, config_channel],
		'worker_id': worker_id,
	}
	return UnityEnvironment(get_env_path(config), **kwargs)
		

# TODO: make this function take in actual params not just the config
def make_env(config, worker_id=0):
	try: unity_env.close()
	except: pass
	try: env.close()
	except: pass
	
	unity_env = create_unity_env(config, worker_id=worker_id)
	
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


'''
# idk y but self.locals['dones'] is always false
class MyCallback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""

	def __init__(self, verbose=2):
		super().__init__(verbose)
		self.rollout_partial_cum_batch_reward = np.zeros(24)

	def _on_step(self) -> bool:
		self.rollout_partial_cum_batch_reward += self.locals['rewards']
				
		avg_cum_batch_reward = 0
		for cum_reward in self.rollout_partial_cum_batch_reward:
			avg_cum_batch_reward += cum_reward
		avg_cum_batch_reward /= 24
		self.logger.record("train/avg_cum_batch_reward", avg_cum_batch_reward)
		
		if all(self.locals['dones']):
			print('Episode Complete')
			print('self.num_timesteps:', self.num_timesteps)
			print('self.num_timesteps / 24:', self.num_timesteps / 24)
			print('avg_cum_batch_reward:', avg_cum_batch_reward)

			# The current episode is over
			self.logger.record("episode/avg_cum_batch_reward", avg_cum_batch_reward)
			self.rollout_partial_cum_batch_reward = np.zeros(24)	# reset for the next rollout

		elif any(self.locals['dones']):
			raise Exception('Black Death is likely not activated. Use SuperSuit Black Death wrapper.')
		
		return True
#'''


class MyCallback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""

	def __init__(self, verbose=2):
		super().__init__(verbose)
		self.past_1k_steps_cum_batch_reward = np.zeros(24)
		self.last_log_at_step = 0

	def _on_step(self) -> bool:
		self.past_1k_steps_cum_batch_reward += self.locals['rewards']
				
		avg_cum_batch_reward = 0
		for cum_reward in self.past_1k_steps_cum_batch_reward:
			avg_cum_batch_reward += cum_reward
		avg_cum_batch_reward /= 24
		
		if self.num_timesteps - self.last_log_at_step >= 1000:
			self.logger.record("train/past_1k_steps_cum_batch_reward", avg_cum_batch_reward)
			self.logger.record("train/cur_step_rewards", self.locals['rewards'])
			self.past_1k_steps_cum_batch_reward = np.zeros(24)	# reset for the next rollout
			self.last_log_at_step = self.num_timesteps
		
		return True


class Eval_every_k_steps_during_training(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""
	def __init__(self, eval_env, eval_every_n_steps: int, eval_ep_batch_limit, verbose=2):
		super().__init__(verbose)
		self.eval_env = eval_env
		self.eval_every_n_steps = eval_every_n_steps
		self.past_k_steps_cum_batch_reward = np.zeros(24)
		self.last_log_at_step = 0
		self.is_pruned = False
		self.eval_ep_batch_limit = eval_ep_batch_limit

	def _on_step(self) -> bool:
		self.past_k_steps_cum_batch_reward += self.locals['rewards']
		
		avg_cum_batch_reward = 0
		for cum_reward in self.past_k_steps_cum_batch_reward:
			avg_cum_batch_reward += cum_reward
		avg_cum_batch_reward /= 24
		
		self.logger.record("train/cur_step_rewards", self.locals['rewards'])
		wandb.log({"train/cur_step_rewards": self.locals['rewards']})
		
		if (self.num_timesteps - self.last_log_at_step) / 24 >= self.eval_every_n_steps:
			print('\n-------------')
			print('Evaluating')
			print('self.num_timesteps:', self.num_timesteps)
			print('self.eval_every_n_steps:', self.eval_every_n_steps)
			
			self.logger.record("train/past_k_steps_cum_batch_reward", avg_cum_batch_reward)
			wandb.log({"train/avg_cum_batch_reward": avg_cum_batch_reward})
			self.past_k_steps_cum_batch_reward = np.zeros(24)	# reset for the next rollout
			self.last_log_at_step = self.num_timesteps

			eval_ep_reward_means, eval_ep_lens = evaluate_policy(self.model, self.eval_env, n_eval_episodes=24 * self.eval_ep_batch_limit, return_episode_rewards=True)
			
			print('eval/reward_means:', eval_ep_reward_means)
			eval_ep_reward_means_mean = statistics.mean(eval_ep_reward_means)
			eval_ep_reward_means_std = statistics.stdev(eval_ep_reward_means)
			print('eval/reward_means_mean:', eval_ep_reward_means_mean)
			print('eval/reward_means_std:', eval_ep_reward_means_std)
		
			wandb.log({
				'eval_ep_reward_means': eval_ep_reward_means,
				'eval_ep_reward_means_mean': eval_ep_reward_means_mean,
				'eval_ep_reward_means_std': eval_ep_reward_means_std,
			})
			print('-------------\n')


class Multi_Agent_Eval_During_Training(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""

	def __init__(self, eval_env, episode_length, eval_every_n_steps, eval_ep_batch_limit, verbose=2):
		'''
		NOTE: eval_every_n_steps is the number of steps (this is self.num_timesteps / n_envs) not the number of steps taken by all agents
		'''
		super().__init__(verbose)
		self.eval_env = eval_env
		self.episode_length = episode_length
		self.eval_every_n_steps = eval_every_n_steps
		self.rollout_partial_cum_batch_reward = np.zeros(24)
		self.last_ep_time_step = 0
		self.eval_ep_batch_limit = eval_ep_batch_limit

	def _on_step(self) -> bool:
		self.rollout_partial_cum_batch_reward += self.locals['rewards']

		avg_cum_batch_reward = 0
		for cum_reward in self.rollout_partial_cum_batch_reward:
			avg_cum_batch_reward += cum_reward
		avg_cum_batch_reward /= 24
		self.logger.record("train/avg_cum_batch_reward", avg_cum_batch_reward)
		wandb.log({"train/avg_cum_batch_reward": avg_cum_batch_reward})
		
		# we divide by 24 because self.num_timesteps is the total number of steps taken by all agents
		if (self.num_timesteps - self.last_ep_time_step) / 24 >= self.episode_length:
			print('Episode Complete')
			print('self.num_timesteps:', self.num_timesteps)
			print('self.num_timesteps / 24:', self.num_timesteps / 24)
			print('avg_cum_batch_reward:', avg_cum_batch_reward)

			# The current episode is over
			self.logger.record("episode/avg_cum_batch_reward", avg_cum_batch_reward)
			wandb.log({"episode/avg_cum_batch_reward": avg_cum_batch_reward})
			self.rollout_partial_cum_batch_reward = np.zeros(24)	# reset for the next rollout

			self.last_ep_time_step = self.num_timesteps

		if (self.num_timesteps / 24) % self.eval_every_n_steps == 0:
			print('\n-------------')
			print('Evaluating')
			print('self.num_timesteps:', self.num_timesteps)
			print('self.eval_every_n_steps:', self.eval_every_n_steps)
			
			eval_ep_reward_means = nasty_evaluate_policy(self.model, self.eval_env, episode_length=self.episode_length, episode_batch_limit=self.eval_ep_batch_limit)
			
			print('eval/reward_means:', eval_ep_reward_means)
			eval_ep_reward_means_mean = statistics.mean(eval_ep_reward_means)
			eval_ep_reward_means_std = statistics.stdev(eval_ep_reward_means)
			print('eval/reward_means_mean:', eval_ep_reward_means_mean)
			print('eval/reward_means_std:', eval_ep_reward_means_std)
		
			wandb.log({
				'eval_ep_reward_means': eval_ep_reward_means,
				'eval_ep_reward_means_mean': eval_ep_reward_means_mean,
				'eval_ep_reward_means_std': eval_ep_reward_means_std,
			})
			print('-------------\n')


def sb_training(config, base_worker_id=0):
	run = wandb.init(
		project="ReMEMber",
		config=config,
		sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
		monitor_gym=False,  # auto-upload the videos of agents playing the game
		save_code=False,  # optional
		#mode='disabled'	# this makes it so nothing is logged and useful to avoid logging debugging runs
	)

	env = make_env(config, worker_id= base_worker_id)
	eval_env = make_env(config, worker_id= base_worker_id +1)

	#algo_hparams = get_tuned_hparams(config)
	algo_hparams = {}
	config['algo_hparams'] = algo_hparams
	wandb.config.update(config)

	algo = get_algo(config['algo_name'])
	model = algo(config["policy_type"], env, verbose=config['verbosity'], tensorboard_log=f"runs/{run.id}", **algo_hparams)
	#model = algo(config["policy_type"], env, verbose=config['verbosity'], tensorboard_log=f"runs/{run.id}")
	
	wandb_callback = WandbCallback(
				gradient_save_freq=100,
				model_save_path=f"models/{run.id}",
				verbose=config['verbosity'],
			)
	
	if config['parallelism'] == 'single_agent':
		
		eval_callback = Eval_every_k_steps_during_training(
			eval_env=eval_env,
			eval_every_n_steps=config['eval_every_n_steps'],
			eval_ep_batch_limit=config['eval_ep_batch_limit']
		)
	
	elif config['parallelism'] == 'multi_agent':
		# importing here to avoid circular import
		#from optuna_hparam_search_multi import TRAIN_Per_Episode_Callback
		#train_callback = TRAIN_Per_Episode_Callback(None, get_episode_length(config['env_name']), ignore_trial=True)

		eval_callback = Multi_Agent_Eval_During_Training(
												eval_env=eval_env,
												episode_length=config['task_configs']['episode_step_count'],
												eval_every_n_steps=config['eval_every_n_steps'],
												eval_ep_batch_limit=config['eval_ep_batch_limit'])

	model.learn(
		total_timesteps=config["total_timesteps"],
		callback=[wandb_callback, eval_callback],
		progress_bar=True
	)
		
	'''
	eval_ep_reward_means_mean = statistics.mean(eval_ep_reward_means)
	eval_ep_reward_means_std = statistics.stdev(eval_ep_reward_means)
	print('eval_ep_reward_means:', eval_ep_reward_means)
	print('eval_ep_reward_means_mean:', eval_ep_reward_means_mean)
	print('eval_ep_reward_means_std:', eval_ep_reward_means_std)

	wandb.log({
		'eval_ep_reward_means': eval_ep_reward_means,
		'eval_ep_reward_means_mean': eval_ep_reward_means_mean,
		'eval_ep_reward_means_std': eval_ep_reward_means_std,
	})
	#'''

	wandb.log({'trial_completed': True})

	env.close()
	run.finish()


# TODO: make this function take in actual params not just the config
def get_tuned_hparams(config):
	'''
	RecurentPPO:
		- exponent_n_steps -> 2 ** n_steps
		- batch_size -> 2 ** batch_size
		- lr -> learning_rate
	
	PPO:
		- delete net_arch
		- place activation_fn in policy_kwargs dict
	
	A2C:
		- delete net_arch
		- delete lr_schedule
		- place otho_init and activation_fn in policy_kwargs dict
	
	DQN:
		- delete net_arch
		- delete subsample_steps
	'''

	if config['env_name'] == 'AllergicRobot':
		if config['algo_name'] == 'RecurrentPPO':
			return {'gamma': 0.05162942889076221, 'max_grad_norm': 0.3057031072878, 'gae_lambda': 0.048305110032528104, 'n_steps': 2**11, 'batch_size': 2**4, 'learning_rate': 2.8228513595556785e-05}
		elif config['algo_name'] == 'PPO':
			return {'batch_size': 64, 'n_steps': 1024, 'gamma': 0.99, 'learning_rate': 0.0003220875664345723, 'ent_coef': 0.022499613441502295, 'clip_range': 0.2, 'n_epochs': 1, 'gae_lambda': 0.9, 'max_grad_norm': 0.5, 'vf_coef': 0.6571489857204025, 'policy_kwargs': {'activation_fn': 'relu'}}
		elif config['algo_name'] == 'A2C':
			return {'gamma': 0.9999, 'normalize_advantage': True, 'max_grad_norm': 1, 'use_rms_prop': True, 'gae_lambda': 0.8, 'n_steps': 256, 'learning_rate': 0.00031943120803574587, 'ent_coef': 0.007825130002826634, 'vf_coef': 0.18499334787277233, 'policy_kwargs': {'ortho_init': True, 'activation_fn': 'relu'}}
		elif config['algo_name'] == 'DQN':
			return {'gamma': 0.95, 'learning_rate': 2.5403375669292505e-05, 'batch_size': 32, 'buffer_size': 50000, 'exploration_final_eps': 0.014933120004429902, 'exploration_fraction': 0.1074899271846918, 'target_update_interval': 10000, 'learning_starts': 1000, 'train_freq': 8}

	elif config['env_name'] == 'MatchingPairs':
		if config['algo_name'] == 'RecurrentPPO':
			return {'gamma': 0.00018762999357827245, 'max_grad_norm': 0.7997837266556623, 'gae_lambda': 0.011356657047837546, 'n_steps': 2**9, 'batch_size': 2**8, 'learning_rate': 0.0024636321465350977}
		elif config['algo_name'] == 'PPO':
			return {'batch_size': 128, 'n_steps': 8, 'gamma': 0.995, 'learning_rate': 0.06306895058245429, 'ent_coef': 0.0024140577982147264, 'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.99, 'max_grad_norm': 0.5, 'vf_coef': 0.8982249321828276, 'policy_kwargs': {'activation_fn': 'tanh'}}
		elif config['algo_name'] == 'A2C':
			return {'gamma': 0.9, 'normalize_advantage': False, 'max_grad_norm': 0.9, 'use_rms_prop': True, 'gae_lambda': 0.99, 'n_steps': 8, 'learning_rate': 0.0009614056590056587, 'ent_coef': 0.0004704211381462, 'vf_coef': 0.6766379244656838, 'policy_kwargs': {'ortho_init': True, 'activation_fn': 'tanh'}}
		elif config['algo_name'] == 'DQN':
			return {'gamma': 0.98, 'learning_rate': 3.231135953900954e-05, 'batch_size': 64, 'buffer_size': 50000, 'exploration_final_eps': 0.10192608886462212, 'exploration_fraction': 0.3766475218948329, 'target_update_interval': 15000, 'learning_starts': 10000, 'train_freq': 8}

	elif config['env_name'] == 'RecipeRecall':
		if config['algo_name'] == 'RecurrentPPO':
			return {'gamma': 0.0010718881282612545, 'max_grad_norm': 0.9772450706550261, 'gae_lambda': 0.035433708191283886, 'n_steps': 2**12, 'batch_size': 2**4, 'learning_rate': 1.765747016242211e-05}
		elif config['algo_name'] == 'PPO':
			return {'batch_size': 256, 'n_steps': 16, 'gamma': 0.99, 'learning_rate': 0.0015797642791422666, 'ent_coef': 0.09811609107275152, 'clip_range': 0.3, 'n_epochs': 20, 'gae_lambda': 0.8, 'max_grad_norm': 0.8, 'vf_coef': 0.1401427820809994, 'policy_kwargs': {'activation_fn': 'tanh'}}
		elif config['algo_name'] == 'A2C':
			return {'gamma': 0.995, 'normalize_advantage': False, 'max_grad_norm': 0.8, 'use_rms_prop': True, 'gae_lambda': 0.8, 'n_steps': 512, 'learning_rate': 0.8745687561393394, 'ent_coef': 4.69458776891305e-08, 'vf_coef': 0.1401427820809994, 'policy_kwargs': {'ortho_init': False, 'activation_fn': 'tanh'}}
		elif config['algo_name'] == 'DQN':
			return {'gamma': 0.98, 'learning_rate': 0.00036789267373174057, 'batch_size': 128, 'buffer_size': 50000, 'exploration_final_eps': 0.02244382854142717, 'exploration_fraction': 0.4265962665550733, 'target_update_interval': 10000, 'learning_starts': 0, 'train_freq': 256}

	elif config['env_name'] == 'Hallway':
		if config['algo_name'] == 'RecurrentPPO':
			return {'gamma': 0.008874973671221307, 'max_grad_norm': 0.8243599176209708, 'gae_lambda': 0.001263813978827776, 'n_steps': 2**8, 'batch_size': 2**9, 'learning_rate': 0.00010856854224558122}
		elif config['algo_name'] == 'PPO':
			return {'batch_size': 16, 'n_steps': 128, 'gamma': 0.995, 'learning_rate': 0.00002829867046740601, 'ent_coef': 0.024572636179435677, 'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.92, 'max_grad_norm': 5, 'vf_coef': 0.02736370912859387, 'policy_kwargs': {'activation_fn': 'tanh'}}
		elif config['algo_name'] == 'A2C':
			return {'gamma': 0.95, 'normalize_advantage': False, 'max_grad_norm': 0.5, 'use_rms_prop': True, 'gae_lambda': 0.8, 'n_steps': 256, 'learning_rate': 0.00002380358967013441, 'ent_coef': 0.00000203659391223045, 'vf_coef': 0.9846257776254022, 'policy_kwargs': {'ortho_init': False, 'activation_fn': 'tanh'}}
		elif config['algo_name'] == 'DQN':
			return {'gamma': 0.9, 'learning_rate': 0.0004352354146364453, 'batch_size': 512, 'buffer_size': 1_000_000, 'exploration_final_eps': 0.09268838794133678, 'exploration_fraction': 0.2361913661788495, 'target_update_interval': 5000, 'learning_starts': 1000, 'train_freq': 1000}
		
	elif config['env_name'] == 'NighttimeNibble':
		if config['algo_name'] == 'RecurrentPPO':
			return {'gamma': 0.00016363606806014624, 'max_grad_norm': 0.5924896508351762, 'gae_lambda': 0.0352077488139621, 'n_steps': 2**10, 'batch_size': 2**4, 'learning_rate': 1.5447298394759683e-05}
		elif config['algo_name'] == 'PPO':
			return {'batch_size': 16, 'n_steps': 1024, 'gamma': 0.9, 'learning_rate': 0.00022179549930824454, 'ent_coef': 0.0007748112012025851, 'clip_range': 0.3, 'n_epochs': 5, 'gae_lambda': 0.98, 'max_grad_norm': 0.5, 'vf_coef': 0.4131187535782552, 'policy_kwargs': {'activation_fn': 'relu'}}
		elif config['algo_name'] == 'A2C':
			return {'gamma': 0.95, 'normalize_advantage': False, 'max_grad_norm': 2, 'use_rms_prop': True, 'gae_lambda': 0.8, 'n_steps': 128, 'learning_rate': 6.199039493203095e-05, 'ent_coef': 0.07967237805050456, 'vf_coef': 0.46295090640468417, 'policy_kwargs': {'ortho_init': False, 'activation_fn': 'relu'}}
		elif config['algo_name'] == 'DQN':
			return {'gamma': 0.99, 'learning_rate': 1.302261230213109e-05, 'batch_size': 512, 'buffer_size': 1000000, 'exploration_final_eps': 0.08855880994513149, 'exploration_fraction': 0.29743629876761535, 'target_update_interval': 1000, 'learning_starts': 10000, 'train_freq': 1000}

	raise Exception('No tuned hparams found')


# TODO: make this function take in actual params not just the config
def get_env_path(config):
	if config['parallelism'] == 'multi_agent':
		parallelism_folder = 'multi_agent'
	elif config['parallelism'] == 'single_agent':
		parallelism_folder = 'single_agent'
	else:
		raise ValueError()

	if config['os'] == 'linux':
		return f'{PROJECT_DIR_PATH}/Builds/{config["env_name"]}/{config["task_variant"]}/linux/pixel_input/{parallelism_folder}/gamefile.x86_64'
	elif config['os'] == 'windows':
		return f'{PROJECT_DIR_PATH}/Builds/{config["env_name"]}/{config["task_variant"]}/windows/pixel_input/{parallelism_folder}/memory_palace_2.exe'
	else:
		raise ValueError()


def get_algo(algo_name):
	algos = {
		'RecurrentPPO': RecurrentPPO,
		'PPO': PPO,
		'SAC': SAC,
		'A2C': A2C,
		'DQN': DQN,
	}
	return algos[algo_name]


def get_default_episode_length(task_name):
	ep_lens = {
		'AllergicRobot': 60,	#60 steps for each of our 24 agents
		'MatchingPairs': 300,
		'RecipeRecall': 80,
		'NighttimeNibble': 80,
		#Hallway has a time limit of 500 steps
	}
	return ep_lens[task_name]


if __name__ == '__main__':
	# load config given cmd line arg and launch training

	print('sys.argv:', sys.argv)
	config_path = sys.argv[1]
	print("CONFIG PATH................", config_path)

	with open(config_path, 'r') as f:
		config = json.load(f)

	sb_training(config=config)