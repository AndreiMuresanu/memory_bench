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
	for config_name, config_value in config['task_configs'].items():
		setup_channel.set_float_parameter(config_name, config_value)

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
	def __init__(self, eval_env, eval_every_n_steps: int, verbose=2):
		super().__init__(verbose)
		self.eval_env = eval_env
		self.eval_every_n_steps = eval_every_n_steps
		self.past_k_steps_cum_batch_reward = np.zeros(24)
		self.last_log_at_step = 0
		self.is_pruned = False

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

			eval_ep_reward_means, eval_ep_lens = evaluate_policy(self.model, self.eval_env, n_eval_episodes=24, return_episode_rewards=True)
			
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

	def __init__(self, eval_env, episode_length, eval_every_n_steps, verbose=2):
		'''
		NOTE: eval_every_n_steps is the number of steps (this is self.num_timesteps / n_envs) not the number of steps taken by all agents
		'''
		super().__init__(verbose)
		self.eval_env = eval_env
		self.episode_length = episode_length
		self.eval_every_n_steps = eval_every_n_steps
		self.rollout_partial_cum_batch_reward = np.zeros(24)
		self.last_ep_time_step = 0

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
			
			eval_ep_reward_means = nasty_evaluate_policy(self.model, self.eval_env, episode_length=self.episode_length, episode_batch_limit=1)
			
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


def sb_training(config):
	run = wandb.init(
		project="ReMEMber",
		config=config,
		sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
		monitor_gym=True,  # auto-upload the videos of agents playing the game
		save_code=True,  # optional
		#mode='disabled'	# this makes it so nothing is logged and useful to avoid logging debugging runs
	)

	env = make_env(config, worker_id= 10 * config['trial_num'] + 1)
	eval_env = make_env(config, worker_id= 10 * config['trial_num'] + 2)

	algo_hparams = get_tuned_hparams(config)
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
			eval_every_n_steps=config['eval_every_n_steps']
		)
	
	elif config['parallelism'] == 'multi_agent':
		# importing here to avoid circular import
		#from optuna_hparam_search_multi import TRAIN_Per_Episode_Callback
		#train_callback = TRAIN_Per_Episode_Callback(None, get_episode_length(config['env_name']), ignore_trial=True)

		eval_callback = Multi_Agent_Eval_During_Training(
												eval_env=eval_env,
												episode_length=get_episode_length(config['env_name']),
												eval_every_n_steps=config['eval_every_n_steps'])

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

	env.close()
	run.finish()


# TODO: make this function take in actual params not just the config
def get_tuned_hparams(config):
	'''
	RecurentPPO:
		- exponent_n_steps -> 2 ** n_steps, 2 ** batch_size
		- lr -> learning_rate
	
	PPO:
		- delete net_arch
		- place otho_init and activation_fn in policy_kwargs dict
	
	A2C:
		- delete net_arch
		- delete lr_schedule
		- place otho_init and activation_fn in policy_kwargs dict
	
	DQN:
		- delete net_arch
		- remove subsample_steps
	'''

	if config['env_name'] == 'MatchingPairs':
		if config['algo_name'] == 'RecurrentPPO':
			return {'gamma': 0.016746355949932314, 'max_grad_norm': 0.6603616944130661, 'gae_lambda': 0.002969712124408621, 'learning_rate': 0.7938195135784231, 'n_steps': 2 ** 8, 'batch_size': 2 ** 8}
		elif config['algo_name'] == 'PPO':
			return {'batch_size': 512, 'n_steps': 8, 'gamma': 0.995, 'learning_rate': 0.0002135829334270177, 'ent_coef': 0.00036969445332778173, 'clip_range': 0.3, 'n_epochs': 1, 'gae_lambda': 0.9, 'max_grad_norm': 0.3, 'vf_coef': 0.45197572631163613,
	   				'policy_kwargs': {'activation_fn': 'tanh'}}
		elif config['algo_name'] == 'A2C':
			return {'gamma': 0.9, 'normalize_advantage': True, 'max_grad_norm': 2, 'use_rms_prop': True, 'gae_lambda': 0.99, 'n_steps': 2048, 'learning_rate': 0.0030535205881154783, 'ent_coef': 0.0073466987282892054, 'vf_coef': 0.7640759166335485,
	   				'policy_kwargs': {'ortho_init': False, 'activation_fn': 'relu'}}
		elif config['algo_name'] == 'DQN':
			return {'gamma': 0.9999, 'learning_rate': 9.647196829315387e-05, 'batch_size': 256, 'buffer_size': 50000, 'exploration_final_eps': 0.19850177873296748, 'exploration_fraction': 0.4421708361590681, 'target_update_interval': 5000, 'learning_starts': 5000, 'train_freq': 128}

	elif config['env_name'] == 'RecipeRecall':
		if config['algo_name'] == 'RecurrentPPO':
			return {'gamma': 0.0002463424821379382, 'max_grad_norm': 0.408005343907218, 'gae_lambda': 0.0010764759356989878, 'n_steps': 2 ** 11, 'batch_size': 2 ** 11, 'learning_rate': 0.02248448303730574}
		elif config['algo_name'] == 'PPO':
			return {'batch_size': 512, 'n_steps': 512, 'gamma': 0.995, 'learning_rate': 0.8934617169216622, 'ent_coef': 2.2607764181947015e-08, 'clip_range': 0.2, 'n_epochs': 1, 'gae_lambda': 0.99, 'max_grad_norm': 2, 'vf_coef': 0.023592547432844113, 
	   				'policy_kwargs': {'activation_fn': 'tanh'}}
		elif config['algo_name'] == 'A2C':
			return {'gamma': 0.999, 'normalize_advantage': False, 'max_grad_norm': 0.7, 'use_rms_prop': False, 'gae_lambda': 0.8, 'n_steps': 16, 'learning_rate': 0.0028375309829371128, 'ent_coef': 5.844594453589111e-05, 'vf_coef': 0.18602480535962188,
	   				'policy_kwargs': {'ortho_init': False, 'activation_fn': 'tanh'}}
		elif config['algo_name'] == 'DQN':
			return {'gamma': 0.995, 'learning_rate': 0.00010977412517896284, 'batch_size': 256, 'buffer_size': 10000, 'exploration_final_eps': 0.12543413026880715, 'exploration_fraction': 0.20864568101507242, 'target_update_interval': 1, 'learning_starts': 0, 'train_freq': 1000}
	
	elif config['env_name'] == 'Hallway':
		if config['algo_name'] == 'RecurrentPPO':
			return {'gamma': 0.04482441745119327, 'max_grad_norm': 0.5674541225381641, 'gae_lambda': 0.004759636080567537, 'learning_rate': 0.002225787528368454, 'n_steps': 2 ** 12, 'batch_size': 2 ** 12}
		elif config['algo_name'] == 'PPO':
			return {'batch_size': 8, 'n_steps': 1024, 'gamma': 0.9, 'learning_rate': 0.0007235657689826623, 'ent_coef': 3.824480674854831e-07, 'clip_range': 0.4, 'n_epochs': 10, 'gae_lambda': 0.92, 'max_grad_norm': 5, 'vf_coef': 0.5469145402988822,
	   				'policy_kwargs': {'activation_fn': 'relu'}}
		elif config['algo_name'] == 'A2C':
			return {'gamma': 0.99, 'normalize_advantage': False, 'max_grad_norm': 0.5, 'use_rms_prop': False, 'gae_lambda': 1.0, 'n_steps': 8, 'learning_rate': 1.4828748330328278e-05, 'ent_coef': 0.00018153800413650935, 'vf_coef': 0.6482827665705186,
	   				'policy_kwargs': {'ortho_init': True, 'activation_fn': 'tanh'}}
		elif config['algo_name'] == 'DQN':
			return {'gamma': 0.99, 'learning_rate': 2.233062915762578e-05, 'batch_size': 16, 'buffer_size': 50000, 'exploration_final_eps': 0.04709293769626912, 'exploration_fraction': 0.1989450988054115, 'target_update_interval': 5000, 'learning_starts': 1000, 'train_freq': 8}
	
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


def get_episode_length(task_name):
	ep_lens = {
		'AllergicRobot': 60,	#60 steps for each of our 24 agents
		'MatchingPairs': 300,
		'RecipeRecall': 80,
		'NighttimeNibble': 80,
		#Hallway has a time limit of 500 steps
	}
	return ep_lens[task_name]


if __name__ == '__main__':

	task_names = [
		#('AllergicRobot', {}),
		#('MatchingPairs', {}),
		#('MatchingPairs', {
		#	'max_ingredient_count': 20,
		#	'available_ingredient_count': 10
		#})
		('Hallway', {}),
		#('RecipeRecall', {})
	]

	task_variants = {
		'standard',
		#'partial_observability',
	}

	# can't store the algos directly because we want to be able to directly upload the config dict to wandb
	algo_names = [
		#'RecurrentPPO',
		'PPO',
		#'A2C',
		#'DQN',
	]

	base_config = {
		"policy_type": "CnnPolicy",
		#"total_timesteps": 250_000,
		#"total_timesteps": 10_000,
		"total_timesteps": 1_000_000,
		#"total_timesteps": 500_000,
		"eval_every_n_steps": 4167,
		#"eval_every_n_steps": 42,
		#"verbosity": 0,
		"verbosity": 2,
	}
	
	#num_of_trial_repeats = 5
	#num_of_trial_repeats = 2
	num_of_trial_repeats = 1

	#trial_offset = 0	# this is the number of runs already completed, this will also set the seed
	trial_offset = 5	# this is the number of runs already completed, this will also set the seed
	
	base_config['num_of_trial_repeats'] = num_of_trial_repeats
	base_config['trial_offset'] = trial_offset
	
	for task_settings in tqdm(task_names, desc='tasks completed'):
		for task_variant in tqdm(task_variants, desc='task variants completed'):
			for algo_name in tqdm(algo_names, desc='algos completed'):
				for trial_num in tqdm(range(num_of_trial_repeats), desc='trials completed'):
					config = deepcopy(base_config)	# I think deepcopy is likely not needed
					config['env_name'] = task_settings[0]
					config['task_configs'] = task_settings[1]
					config['task_variant'] = task_variant
					config['algo_name'] = algo_name
					config['trial_num'] = trial_num + trial_offset
					config['seed'] = trial_num + trial_offset
		
					if config['env_name'] == 'Hallway':
						config['parallelism'] = 'single_agent'
					else:
						config['parallelism'] = 'multi_agent'

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
