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

from sb_training import make_env, get_episode_length
from rl_zoo_samplers import sample_ppo_params, sample_rppo_params, sample_a2c_params, sample_dqn_params
import statistics


#PROJECT_DIR_PATH = '/h/mskrt/memory_bench'
PROJECT_DIR_PATH = '/h/andrei/memory_bench'

N_TRIALS = 40
SEED_OFFSET = 0
#N_EVALUATIONS = 50
#N_TIMESTEPS = int(200_000)
N_TIMESTEPS = int(20)
#EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)

# convert all this sloppy code into a factory
print(sys.argv); 
task_name = sys.argv[1]
print("TASK NAME................", task_name)

algo_name = sys.argv[2]
print("ALGO NAME................", algo_name)

worker_id = int(sys.argv[3])
print("worker_id................", worker_id)

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

wandb_kwargs = {
		"project":"ReMEMber",
		"entity": "team-andrei",
		"sync_tensorboard":True,  # auto-upload sb3's tensorboard metrics
		#"monitor_gym":True,  # auto-upload the videos of agents playing the game
		"save_code":True,  # optional
		#"name": "{}_{}_200k-steps".format(task_name, algo_name),
		#'mode': 'disabled'	# this makes it so nothing is logged and useful to avoid logging debugging runs
	}
#wandbc = WeightsAndBiasesCallback(metric_name="mean reward", wandb_kwargs=wandb_kwargs, as_multirun=True)
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

base_config = {
	"policy_type": "CnnPolicy",
	#"parallelism": "single_agent",
	"parallelism": "multi_agent",
	
	"total_timesteps": N_TIMESTEPS,
	'N_TRIALS': N_TRIALS,
	'SEED_OFFSET': SEED_OFFSET,

	'env_name': task_name,
	'algo_name': algo_name,
}



class Past_1k_Steps_Callback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""
	def __init__(self, trial: optuna.Trial, verbose=2):
		super().__init__(verbose)
		self.past_1k_steps_cum_batch_reward = np.zeros(24)
		self.last_log_at_step = 0
		self.trial = trial

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
		if self.trial.should_prune():
			return False	
		else:
			return True


class Per_Episode_Callback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""

	def __init__(self, trial: optuna.Trial, episode_length=None, episode_batch_limit=None, result_container=None, verbose=2):
		super().__init__(verbose)
		self.trial = trial
		self.episode_length = episode_length
		self.episode_batch_limit = episode_batch_limit
		self.result_container = result_container
		self.rollout_partial_cum_batch_reward = np.zeros(24)
		self.last_ep_time_step = 0
		self.episodes_seen = 0

	def _on_step(self) -> bool:
		self.rollout_partial_cum_batch_reward += self.locals['rewards']
				
		avg_cum_batch_reward = 0
		for cum_reward in self.rollout_partial_cum_batch_reward:
			avg_cum_batch_reward += cum_reward
		avg_cum_batch_reward /= 24
		self.logger.record("train/avg_cum_batch_reward", avg_cum_batch_reward)
		
		if self.num_timesteps - self.last_ep_time_step >= self.episode_length:
			print('Episode Complete')
			print('self.num_timesteps:', self.num_timesteps)
			print('self.num_timesteps / 24:', self.num_timesteps / 24)
			print('avg_cum_batch_reward:', avg_cum_batch_reward)

			# The current episode is over
			#self.logger.record("episode/avg_cum_batch_reward", avg_cum_batch_reward)
			if self.result_container: self.result_container.append(avg_cum_batch_reward)
			self.rollout_partial_cum_batch_reward = np.zeros(24)	# reset for the next rollout
			self.episodes_seen += 1
			if self.episode_batch_limit and self.episodes_seen >= self.episode_batch_limit:
				return False

			self.last_ep_time_step = self.num_timesteps
		
		# Prune trial if need.
		if self.trial.should_prune():
			return False	
		else:
			return True


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
		model.learn(
			total_timesteps=config["total_timesteps"],
			callback=Past_1k_Steps_Callback(trial),
			progress_bar=True
		)

		if config['parallelism'] == 'single_agent':
			eval_ep_reward_means, eval_ep_lens = evaluate_policy(model, env, n_eval_episodes=24, return_episode_rewards=True, callback=Past_1k_Steps_Callback(trial))
		
		elif config['parallelism'] == 'multi_agent':
			
			eval_ep_reward_means = []	
			eval_callback = Per_Episode_Callback(trial, get_episode_length(config['env_name']), episode_batch_limit=1)
			#eval_callback = Per_Episode_Callback(trial, get_episode_length(config['env_name']), episode_batch_limit=1, result_container=eval_ep_reward_means)
			_, eval_ep_lens = evaluate_policy(model, env, n_eval_episodes=24, return_episode_rewards=True, callback=eval_callback)
		
		else:
			raise ValueError(f"Invalid parallelism: {config['parallelism']}")

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
	
	if eval_callback.is_pruned:
		raise optuna.exceptions.TrialPruned()
	
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

				
