from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import torch.nn as nn
import sys
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import optuna
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from PIL import Image
import os

import gymnasium as gym
import numpy as np
#import matplotlib.pyplot as plt

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
from stable_baselines3.common.evaluation import evaluate_policy

import supersuit as ss
from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
from sb3_contrib import RecurrentPPO

import wandb
#from wandb.integration.sb3 import WandbCallback
from optuna.integration.wandb import WeightsAndBiasesCallback

from copy import deepcopy
from tqdm import tqdm

from sb_training import make_env, get_default_episode_length
from rl_zoo_samplers import sample_ppo_params, sample_rppo_params, sample_a2c_params, sample_dqn_params
import statistics
from nasty_evaluate_policy import nasty_evaluate_policy

#PROJECT_DIR_PATH = '/h/mskrt/memory_bench'
PROJECT_DIR_PATH = '/h/andrei/memory_bench'

#N_TRIALS = 39
#SEED_OFFSET = -1
#N_EVALUATIONS = 49
#N_TIMESTEPS = int(199_000)
N_TIMESTEPS = int(250_000)
#EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)


# would be really nice to move this outside of the global space
wandb_kwargs = {
		"project":"ReMEMber",
		"entity": "team-andrei",
		"sync_tensorboard":True,  # auto-upload sb3's tensorboard metrics
		#"monitor_gym":True,  # auto-upload the videos of agents playing the game
		#"save_code":True,  # optional
		#"name": "{}_{}_200k-steps".format(task_name, algo_name),
		#'mode': 'disabled'	# this makes it so nothing is logged and useful to avoid logging debugging runs
	}
#wandbc = WeightsAndBiasesCallback(metric_name="mean reward", wandb_kwargs=wandb_kwargs, as_multirun=True)
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)



class Past_1k_Steps_Callback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""
	def __init__(self, trial: optuna.Trial, verbose=2):
		super().__init__(verbose)
		self.past_1k_steps_cum_batch_reward = np.zeros(24)
		self.last_log_at_step = 0
		self.trial = trial
		self.is_pruned = False

	def _on_step(self) -> bool:
		self.past_1k_steps_cum_batch_reward += self.locals['rewards']
		
		avg_cum_batch_reward = 0
		for cum_reward in self.past_1k_steps_cum_batch_reward:
			avg_cum_batch_reward += cum_reward
		avg_cum_batch_reward /= 24
		
		self.logger.record("train/cur_step_rewards", self.locals['rewards'])
		wandb.log({"train/cur_step_rewards": self.locals['rewards']})
		
		if self.num_timesteps - self.last_log_at_step >= 1000:
			self.logger.record("train/past_1k_steps_cum_batch_reward", avg_cum_batch_reward)
			wandb.log({"train/avg_cum_batch_reward": avg_cum_batch_reward})
			self.past_1k_steps_cum_batch_reward = np.zeros(24)	# reset for the next rollout
			self.last_log_at_step = self.num_timesteps
		
		# Prune trial if need.
		if self.trial and self.trial.should_prune():
			self.is_pruned = True
			return False	
		else:
			return True


class TRAIN_Per_Episode_Callback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""

	def __init__(self, trial: optuna.Trial, episode_length, verbose=2, ignore_trial=False):
		super().__init__(verbose)
		self.trial = trial
		self.episode_length = episode_length
		self.ignore_trial = ignore_trial
		self.rollout_partial_cum_batch_reward = np.zeros(24)
		self.last_ep_time_step = 0
		self.is_pruned = False

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
		
		# Prune trial if need.
		if not self.ignore_trial and self.trial.should_prune():
			self.is_pruned = True
			return False
		else:
			return True

'''
class EVAL_Per_Episode_Callback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""

	def __init__(self, trial: optuna.Trial, episode_length, episode_batch_limit, result_container, verbose=2):
		super().__init__(verbose)
		self.trial = trial
		self.episode_length = episode_length
		self.episode_batch_limit = episode_batch_limit
		self.result_container = result_container

		self.rollout_partial_cum_batch_reward = np.zeros(24)
		self.last_ep_time_step = 0
		self.episodes_seen = 0
		self.is_pruned = False
		self.num_of_steps = 0

	def _on_step(self, locals, globals) -> bool:
		def end_task(locals):
			print('ENDING EPISODE')
			locals['dones'] = np.ones(24)
			locals['episode_counts'] = np.repeat(self.episode_batch_limit, 24)
			locals['episode_count_targets'] = np.repeat(self.episode_batch_limit, 24)

		self.num_of_steps += 24
		self.rollout_partial_cum_batch_reward += locals['rewards']

		#print('self.num_of_steps:', self.num_of_steps)
		#print('self.last_ep_time_step:', self.last_ep_time_step)
		#print('self.episode_length:', self.episode_length)
		#print('statistics.mean(self.rollout_partial_cum_batch_reward):', statistics.mean(self.rollout_partial_cum_batch_reward))
		#print('self.episodes_seen:', self.episodes_seen)

		avg_cum_batch_reward = 0
		for cum_reward in self.rollout_partial_cum_batch_reward:
			avg_cum_batch_reward += cum_reward
		avg_cum_batch_reward /= 24
		
		if self.num_of_steps - self.last_ep_time_step >= self.episode_length:
			print('\nEpisode Complete')
			print('self.num_of_steps:', self.num_of_steps)
			print('self.episode_batch_limit:', self.episode_batch_limit)
			print('avg_cum_batch_reward:', avg_cum_batch_reward)
			print('locals["dones"]:', locals["dones"])

			# The current episode is over
			self.rollout_partial_cum_batch_reward = np.zeros(24)	# reset for the next rollout
			self.episodes_seen += 1
			if self.episodes_seen >= self.episode_batch_limit:
				end_task(locals)

			self.last_ep_time_step = self.num_of_steps
		
		# prune trial if need.
		if self.trial.should_prune():
			self.is_pruned = True
			end_task(locals)
		else:
			pass
#'''

'''
class stupid_per_episode_callback(basecallback):
	"""
	custom callback for plotting additional values in tensorboard.
	"""

	def __init__(self, trial: optuna.trial, episode_length=none, episode_batch_limit=none, result_container=none, verbose=2):
		super().__init__(verbose)
		self.trial = trial
		self.episode_length = episode_length
		self.episode_batch_limit = episode_batch_limit
		self.result_container = result_container
		
		if self.result_container is none:
			self.result_container = []
		
		if 'eval_callback_params' in globals():
			if 'episode_length' in eval_callback_params:
				self.episode_length = eval_callback_params['episode_length']
			if 'episode_batch_limit' in eval_callback_params:
				self.episode_batch_limit = eval_callback_params['episode_batch_limit']
			if 'result_container' in eval_callback_params:
				self.result_container = eval_callback_params['result_container']

		print('self.episode_length:', self.episode_length)
		print('self.episode_batch_limit:', self.episode_batch_limit)
		print('self.result_container:', self.result_container)

		self.rollout_partial_cum_batch_reward = np.zeros(24)
		self.last_ep_time_step = 0
		self.episodes_seen = 0
		self.is_pruned = false
		self.num_of_steps = 0

	def _on_step(self, locals=none, globals=none) -> bool:
		self.num_of_steps += 24
		if self.num_timesteps < self.num_of_steps:
			self.num_timesteps = self.num_of_steps

		if locals is not none:
			self.locals = locals
		if globals is not None:
			self.globals = globals

		#print('locals:', locals)
		#print('globals:', globals)

		print('self.num_of_steps:', self.num_of_steps)
		print('self.num_timesteps:', self.num_timesteps)

		self.rollout_partial_cum_batch_reward += self.locals['rewards']
		
		avg_cum_batch_reward = 0
		for cum_reward in self.rollout_partial_cum_batch_reward:
			avg_cum_batch_reward += cum_reward
		avg_cum_batch_reward /= 24
		#self.logger.record("train/avg_cum_batch_reward", avg_cum_batch_reward)
		
		if self.num_timesteps - self.last_ep_time_step >= self.episode_length:
			print('Episode Complete')
			print('self.num_timesteps:', self.num_timesteps)
			print('self.num_timesteps / 24:', self.num_timesteps / 24)
			print('avg_cum_batch_reward:', avg_cum_batch_reward)

			# The current episode is over
			#self.logger.record("episode/avg_cum_batch_reward", avg_cum_batch_reward)
			self.result_container.append(avg_cum_batch_reward)
			self.rollout_partial_cum_batch_reward = np.zeros(24)	# reset for the next rollout
			self.episodes_seen += 1
			if self.episode_batch_limit and self.episodes_seen >= self.episode_batch_limit:
				return False

			self.last_ep_time_step = self.num_timesteps
		
		# Prune trial if need.
		if self.trial.should_prune():
			self.is_pruned = True
			return False	
		else:
			return True
#'''


#def sb_training(config):
@wandbc.track_in_wandb()
def objective(trial: optuna.Trial) -> float:

	config = base_config.copy() 

	config['trial_num'] = trial.number + SEED_OFFSET
	config['seed'] = trial.number + SEED_OFFSET

	if algo_name == 'RecurrentPPO':
		config['policy_type'] = 'CnnLstmPolicy'
	
	print("config", config)
	wandb.config.update(config)

	env = make_env(config)
	
	kwargs = {"policy": config['policy_type'],
			  "env": env,
			  "verbose": 2,
			  #"tensorboard_log": f"runs/{run.id}"
			  }
	#if trial._trial_id == 0:
	#    kwargs.update(trial.system_attrs['fixed_params'])
	#else:
	if "RecurrentPPO" in algo_name:
		kwargs.update(sample_rppo_params(trial))
	elif "PPO" in algo_name:
		kwargs.update(sample_ppo_params(trial, config['parallelism']))
	elif "A2C" in algo_name:
		kwargs.update(sample_a2c_params(trial))
	elif "DQN" in algo_name or "SAC" in algo_name:
		kwargs.update(sample_dqn_params(trial))
		if "SAC" in algo_name:
			del kwargs['max_grad_norm']
	print("kwargs", kwargs)

	#eval_callback = TrialEvalCallback(
	#    env, trial, n_eval_episodes=3, eval_freq=EVAL_FREQ, deterministic=True
	#)

	algo = get_algo(config['algo_name'])
	model = algo(**kwargs)
	nan_encountered = False
	try:
		
		if config['parallelism'] == 'single_agent':
			train_callback = Past_1k_Steps_Callback(trial)
		elif config['parallelism'] == 'multi_agent':
			train_callback = TRAIN_Per_Episode_Callback(trial, config['task_configs']['episode_step_count'])
		else:
			raise ValueError(f'Invalid parallelism: {config["parallelism"]}')

		model.learn(
			total_timesteps=config["total_timesteps"],
			callback=train_callback,
			progress_bar=True
		)
		
		if train_callback.is_pruned:
			raise optuna.exceptions.TrialPruned()

		print('Evaluating... this may take a while.')

		if config['parallelism'] == 'single_agent':
			#my_eval_callback = Past_1k_Steps_Callback
			#eval_ep_reward_means, eval_ep_lens = evaluate_policy(model, env, n_eval_episodes=24, return_episode_rewards=True, callback=my_eval_callback)
			eval_ep_reward_means, eval_ep_lens = evaluate_policy(model, env, n_eval_episodes=24 * config['num_final_eval_batches'], return_episode_rewards=True)
		
		elif config['parallelism'] == 'multi_agent':
			
			'''
			global EVAL_CALLBACK_PARAMS
			EVAL_CALLBACK_PARAMS = {
				'trial': trial,
				'episode_length': get_episode_length(config['env_name']),
				'result_container': [],
				'episode_batch_limit': 1
			}

			my_eval_callback = Per_Episode_Callback(trial)
			#my_eval_callback = Per_Episode_Callback(trial, get_episode_length(config['env_name']), episode_batch_limit=1, result_container=eval_ep_reward_means)
			
			_, eval_ep_lens = evaluate_policy(model, env, n_eval_episodes=24, return_episode_rewards=True, callback=my_eval_callback._on_step)
		
			eval_ep_reward_means = EVAL_CALLBACK_PARAMS['eval_ep_reward_means']
			EVAL_CALLBACK_PARAMS = {}
			#'''

			'''
			eval_ep_reward_means = []
			my_eval_callback = EVAL_Per_Episode_Callback(trial, get_episode_length(config['env_name']), episode_batch_limit=1, result_container=eval_ep_reward_means)
			
			_, eval_ep_lens = evaluate_policy(model, env, n_eval_episodes=24, return_episode_rewards=True, callback=my_eval_callback._on_step)
			#'''

			#TODO: verify that episode_batch_limit actually does something (hopefully the right thing)
			assert 1 == 2
			eval_ep_reward_means = nasty_evaluate_policy(model, env, episode_length=config['task_configs']['episode_step_count'], episode_batch_limit=config['num_final_eval_batches'])

		else:
			raise ValueError(f"Invalid parallelism: {config['parallelism']}")

		print('eval_ep_reward_means:', eval_ep_reward_means)
		eval_ep_reward_means_mean = statistics.mean(eval_ep_reward_means)
		eval_ep_reward_means_std = statistics.stdev(eval_ep_reward_means)
		print('eval_ep_reward_means_mean:', eval_ep_reward_means_mean)
		print('eval_ep_reward_means_std:', eval_ep_reward_means_std)

		wandb.log({
			'eval_ep_reward_means': eval_ep_reward_means,
			'eval_ep_reward_means_mean': eval_ep_reward_means_mean,
			'eval_ep_reward_means_std': eval_ep_reward_means_std,
		})


	except AssertionError as e:
		# Sometimes, random hyperparams can generate NaN.
		print(e)
		nan_encountered = True
	finally:
		# Free memory.
		#model.env.close()
		env.close()
		#run.finish()
	
	# Tell the optimizer that the trial failed.
	if nan_encountered:
		return float("nan")
	
	#return eval_callback.last_mean_reward
	return eval_ep_reward_means_mean


def get_env_path(config):
	if config['parallelism'] == 'multi_agent':
		parallelism_folder = 'multi_agent'
	elif config['parallelism'] == 'single_agent':
		parallelism_folder = 'single_agent'
	else:
		raise ValueError()

	return f'{PROJECT_DIR_PATH}/Builds/{config["env_name"]}/standard/linux/pixel_input/{parallelism_folder}/gamefile.x86_64'


def get_algo(alg):
	algos = {
		'PPO': PPO,
		'RecurrentPPO': RecurrentPPO,
		'SAC': SAC,
		'A2C': A2C,
		'DQN': DQN,
	}
	return algos[alg]


if __name__ == '__main__':
	# convert all this sloppy code into a factory
	print(sys.argv); 
	task_name = sys.argv[1]
	print("TASK NAME................", task_name)

	algo_name = sys.argv[2]
	print("ALGO NAME................", algo_name)

	worker_id = int(sys.argv[3])
	print("worker_id................", worker_id)

	global SEED_OFFSET
	SEED_OFFSET = int(sys.argv[4])
	print("SEED_OFFSET................", SEED_OFFSET)

	global N_TRIALS
	N_TRIALS = int(sys.argv[5])
	print("N_TRIALS................", N_TRIALS)

	task_names = [
		'AllergicRobot',
		'MatchingPairs',
		'Hallway',
		'RecipeRecall'
	]

	# can't store the algos directly because we want to be able to directly upload the config dict to wandb
	algo_names = [
		'PPO',
		'RecurrentPPO',
		'A2C',
		'DQN',
		#'SAC',
	]

	assert task_name in task_names
	assert algo_name in algo_names	

	base_config = {
		"policy_type": "CnnPolicy",
		#"parallelism": "single_agent",
		#"parallelism": "multi_agent",
		
		"total_timesteps": N_TIMESTEPS,
		'N_TRIALS': N_TRIALS,
		'SEED_OFFSET': SEED_OFFSET,
		'worker_id': worker_id,

		'env_name': task_name,
		'algo_name': algo_name,

		'num_final_eval_batches': 10,	#this corresponds to 100 * 24 eval episodes
		'os': 'linux',
		'task_variant': 'standard',
	}

	#if task_name == 'Hallway':
	#	base_config['parallelism'] = 'single_agent'
	#else:
	#	base_config['parallelism'] = 'multi_agent'
	base_config['parallelism'] = 'single_agent'	# overriding to single_agent parallelism

	base_config['task_configs'] = {}
	if 'episode_step_count' not in base_config['task_configs']:
		base_config['task_configs']['episode_step_count'] = get_default_episode_length(base_config['env_name'])


	# Set pytorch num threads to 1 for faster training.
	torch.set_num_threads(1)

	#sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
	sampler = TPESampler()
	# Do not prune before 1/3 of the max budget is used.
	pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=N_TIMESTEPS // 3)

	study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
	#first_trial_params = {
	#        'learning_rate': 0.0003,
	#        "batch_size": 64,
	#        "n_steps": 2048,
	#}
	#study.enqueue_trial(first_trial_params)
	try:
		#study.optimize(objective, n_trials=N_TRIALS, timeout=600,callbacks=[wandbc])
		study.optimize(objective, n_trials=N_TRIALS, timeout=None,callbacks=[wandbc])
	except KeyboardInterrupt:
		pass

	print("Number of finished trials: ", len(study.trials))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)

	print('study.best_params:', study.best_params)

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))

	print("  User attrs:")
	for key, value in trial.user_attrs.items():
		print("    {}: {}".format(key, value)) 