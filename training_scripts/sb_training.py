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


PROJECT_DIR_PATH = '/h/andrei/memory_bench'


'''
def monkey_SuperSuit_MVW_step(self, actions):
	agent_set = set(self.par_env.agents)
	act_dict = {agent: actions[i] for i, agent in enumerate(self.par_env.possible_agents) if agent in agent_set}
	observations, rewards, dones, infos = self.par_env.step(act_dict)

	print('\nSuperSuit MVP')
	print('observations:', observations)
	print('rewards:', rewards)
	print('dones:', dones)
	print('infos:', infos)

	# adds last observation to info where user can get it
	if all(dones.values()):
		for agent, obs in observations.items():
			infos[agent]['terminal_observation'] = obs

	rews = np.array([rewards.get(agent, 0) for agent in self.par_env.possible_agents], dtype=np.float32)
	dns = np.array([dones.get(agent, False) for agent in self.par_env.possible_agents], dtype=np.uint8)
	infs = [infos.get(agent, {}) for agent in self.par_env.possible_agents]

	if all(dones.values()):
		observations = self.reset()
	else:
		observations = self.concat_obs(observations)
	assert (
		self.black_death or self.par_env.agents == self.par_env.possible_agents
	), "MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True"
	return observations, rews, dns, infs

ss.vector.markov_vector_wrapper.step = monkey_SuperSuit_MVW_step
#'''



'''
# monkey patching to avoid unexpected argument 'seed' error in UnityParallelEnv.reset() 
def monkey_MarkovVectorEnv_reset(self, seed=None, options=None):
	# TODO: should this be changed to infos?
	
	# Function copied from: https://github.com/Farama-Foundation/SuperSuit/blob/master/supersuit/vector/markov_vector_wrapper.py
	# Original:
	_observations, infos = self.par_env.reset(seed=seed, options=options)
	
	# New: 
	#_observations = self.par_env.reset(seed=seed, options=options)

	# New:
	#kwargs = {}
	#if seed: kwargs['seed'] = seed
	#if options: kwargs['options'] = options
	#print(kwargs)
	#print(type(self.par_env))
	#_observations, infos = self.par_env.reset(**kwargs)
	
	observations = self.concat_obs(_observations)
	
	# Original:
	infs = [infos.get(agent, {}) for agent in self.par_env.possible_agents]
	
	# New:
	#infs = [{} for agent in self.par_env.possible_agents]

	return observations, infs

ss.vector.markov_vector_wrapper.MarkovVectorEnv.reset = monkey_MarkovVectorEnv_reset
#'''

'''
# monkey patching to avoid unexpected argument 'seed' error in UnityParallelEnv.reset() 
def reset(self, seed=None, options=None):
	self.aec_env.reset(seed=seed, options=options)
	self.agents = self.aec_env.agents[:]
	observations = {
		agent: self.aec_env.observe(agent)
		for agent in self.aec_env.agents
		if not (self.aec_env.terminations[agent] or self.aec_env.truncations[agent])
	}

	infos = dict(**self.aec_env.infos)
	return observations, infos
#'''


from mlagents_envs.envs.unity_pettingzoo_base_env import UnityPettingzooBaseEnv
#from monkey_unity_pz_env import monkey_UnityPettingzooBaseEnv
#UnityPettingzooBaseEnv = monkey_UnityPettingzooBaseEnv

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.envs.unity_aec_env import UnityAECEnv
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

#'''
# monkey patching to avoid unexpected argument 'seed' error in UnityParallelEnv.reset() 
def monkey_UnityParallelEnv_reset(self, seed=None, options=None):
	"""
	Resets the environment.
	"""
	UnityPettingzooBaseEnv.reset(self)

	#print('self._observations:', self._observations)
	print('type(self._observations):', type(self._observations))

	return self._observations, {}

UnityParallelEnv.reset = monkey_UnityParallelEnv_reset
#'''


'''
# This module is loaded directly from gymnasium before by supersuit before I have a chance to monkey patch it

# TODO: The indicies should return the attr for each env in the vector env.
def monkey_Gymnasium_VectorEnv_get_attr(self, name: str, indices = None):
	"""Get a property from each parallel environment.

	Args:
		name (str): Name of the property to be get from each individual environment.

	Returns:
		The property with name
	"""
	if indices:
		raise Exception('Did not account for this.')

	print('Using monkey get_attr')

	return self.call(name)

gym.vector.VectorEnv.get_attr = monkey_Gymnasium_VectorEnv_get_attr
#'''


def monkey_VecEnvWrapper_get_attr(self, attr_name: str, indices = None):
	print('attr_name:', attr_name)
	print('indices:', indices)
	print('type(self.venv):', type(self.venv))
	if attr_name == 'render_mode':
		raise AttributeError('This is causing issues so we are going to explicitly ignore it.')

	if indices is not None:
		raise Exception('Did not account for this.')

	return self.venv.get_attr(attr_name)

import stable_baselines3
stable_baselines3.common.vec_env.base_vec_env.VecEnvWrapper.get_attr = monkey_VecEnvWrapper_get_attr

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback


#'''
def monkey_SS_concat_obs(self, obs_dict):
	if self.black_death:
		self.obs_buffer[:] = 0
	for i, agent in enumerate(self.par_env.possible_agents):
		if self.black_death:
			if agent in obs_dict:
				#print('===========')
				#print('agent:', agent)
				##print('obs_dict[agent]:', obs_dict[agent])
				#print("obs_dict[agent]['observation']:", obs_dict[agent]['observation'])
				#print('self.obs_buffer[i]:', self.obs_buffer[i])
				#print("len(obs_dict[agent]['observation']):", len(obs_dict[agent]['observation']))
				#print("obs_dict[agent]['observation'][0].shape:", obs_dict[agent]['observation'][0].shape)
				#print('self.obs_buffer[i].shape:', self.obs_buffer[i].shape)

				# ORIGINAL:
				#self.obs_buffer[i] = obs_dict[agent]
				# NEW:
				assert len(obs_dict[agent]['observation']) == 1, 'I don\'t know how this should be handled if there is more then 1 observation'
				self.obs_buffer[i] = obs_dict[agent]['observation'][0]
		else:
			if agent not in obs_dict:
				raise AssertionError("environment has agent death. Not allowed for pettingzoo_env_to_vec_env_v1 unless black_death is True")
			self.obs_buffer[i] = obs_dict[agent]

	return self.obs_buffer.copy()
#'''


import supersuit as ss

from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
from sb3_contrib import RecurrentPPO




def input_checking():
	unity_env = UnityEnvironment('/h/andrei/memory_bench/Builds/AllergicRobot/standard/linux/pixel_input/single_agent/gamefile.x86_64')
	env = UnityToGymWrapper(unity_env, uint8_visual=True)

	observation = env.reset()
	print(observation)
	print(type(observation))
	print('observation.shape:', observation.shape)
	for cur_step in range(30):
		action = env.action_space.sample()  # this is where you would insert your policy
		observation, reward, done, info = env.step(action)

		#agent_view = Image.fromarray(observation)
		#agent_view.save(f'./misc_results/agent_view_{cur_step}.png')
	
		if done: observation = env.reset()
	
	env.close()


def multi_agent_input_checking(config):
	env = make_env(config)

	print(type(env))

	observation = env.reset()
	print(observation[0])
	print(type(observation))
	print('observation.shape:', observation.shape)
	print(type(observation[0]))
	print('observation[0].shape:', observation[0].shape)

	# the below code is wrong and will fail

	for cur_step in range(30):
		action = env.action_space.sample()  # this is where you would insert your policy
		observation, reward, done, info = env.step(action)

		#agent_view = Image.fromarray(observation)
		#agent_view.save(f'./misc_results/agent_view_{cur_step}.png')
	
		if done: observation = env.reset()
	
	env.close()


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
	
	#env = Monitor(env=env)  # record stats such as returns


def make_env(config):
	try: unity_env.close()
	except: pass
	try: env.close()
	except: pass
	
	unity_env = create_unity_env(config)
	
	if config['parallelism'] == 'single_agent':
		return UnityToGymWrapper(unity_env, uint8_visual=True)
	
	elif config['parallelism'] == 'multi_agent':
		
		#env = UnityAECEnv(unity_env)
		#env.metadata = {'is_parallelizable': True}
		#env = aec_to_parallel(env)
		
		# This does not produce errors... until later
		env = UnityParallelEnv(unity_env)
		
		print('env.observation_spaces:', env.observation_spaces)

		for key, value in env._observation_spaces.items():
			#env._observation_spaces[key] = real_gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
			env._observation_spaces[key] = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
		
		print('env.observation_spaces:', env.observation_spaces)
		
		print('\n---------\n')

		#'''
		print('env.action_spaces:', env.action_spaces)
		
		for key, value in env._action_spaces.items():
			env._action_spaces[key] = gym.spaces.discrete.Discrete(7)
		
		print('env.action_spaces:', env.action_spaces)
		#'''
		
		#env = ss.pad_action_space_v0(env)
		#env = ss.pad_observations_v0(env)
		#env = ss.black_death_v2(env)
		
		env = ss.pettingzoo_env_to_vec_env_v1(env)
		
		env.black_death = True	# THIS IS WRONG. We are just suppressing the error. Look here: https://github.com/Farama-Foundation/SuperSuit/blob/master/supersuit/vector/markov_vector_wrapper.py#L38


		#print('\nMONKEY PATCHING CONCAT OBS')
		#print('type(env):', type(env))
		#env.concat_obs = monkey_SS_concat_obs
		#print('DONE MONKEY PATCHING CONCAT OBS\n')


		#env.observation_space = real_gym.spaces.Box(low=0, high=255, shape=(24, 84, 84, 3), dtype=np.uint8)
		#env = ss.dtype_v0(env, dtype=np.uint8)
		#env = ss.dtype_v0(env, dtype=np.float32)
		#env = ss.observation_lambda_v0(env,
		#		 lambda x, y: x,
		#		 lambda obs_space : real_gym.spaces.Box(low=0, high=255, shape=(84, 84, 3)))
		#env = ss.normalize_obs_v0(env, env_min=0, env_max=1)

		#env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
		#env = ss.concat_vec_envs_v1(env, 24, num_cpus=32, base_class='stable_baselines3')

		#env = ss.stable_baselines3_vec_env_v0(env, 1, multiprocessing=False)

		print('type(env):', type(env))
		print('type(env.observation_space):', type(env.observation_space))
		print('env.observation_space:', env.observation_space)
		print('env.num_envs:', env.num_envs)


		#check_env(env)

		env = VecMonitor(venv=env)
		
		print('\nMONKEY PATCHING CONCAT OBS')
		print('type(env.venv):', type(env.venv))
		env.venv.concat_obs = monkey_SS_concat_obs.__get__(env.venv, ss.vector.markov_vector_wrapper.MarkovVectorEnv)
		print('DONE MONKEY PATCHING CONCAT OBS\n')

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
		print('+++++++++++++++++++++++++++++++++++')
		print('ROLLOUT')
		print('+++++++++++++++++++++++++++++++++++')
		print('self.num_timesteps:', self.num_timesteps)
		print('self.rollout_partial_cum_batch_reward:', self.rollout_partial_cum_batch_reward)

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

	print('MAKE ENV COMPLETED')


	print('type(env):', type(env))
	
	#print('env.action_space:', env.action_space)

	print('type(env.venv):', type(env.venv))
	print('type(env.venv.par_env):', type(env.venv.par_env))
	print('')
	print('type(env.venv.par_env._action_spaces):', type(env.venv.par_env._action_spaces))
	print('env.venv.par_env._action_spaces:', env.venv.par_env._action_spaces)
	_, agent0_as = list(env.venv.par_env._action_spaces.items())[0]
	print('type(agent0_as):', type(agent0_as))
	print('agent0_as:', agent0_as)
	print('')
	_, agent0_os = list(env.venv.par_env._observation_spaces.items())[0]
	print('type(agent0_os):', type(agent0_os))
	print('agent0_os:', agent0_os)

	print('------')
	print('env.venv.par_env.possible_agents:', env.venv.par_env.possible_agents)
	print('type(env.venv.par_env.possible_agents[0]):', type(env.venv.par_env.possible_agents[0]))
	print('------')

	#env = DummyVecEnv([make_env(config)])
	#env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

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

	#model = PPO("CnnPolicy", env, verbose=2)
	#model.learn(total_timesteps=3)

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
	#input_checking()
	#assert 1 == 2

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
