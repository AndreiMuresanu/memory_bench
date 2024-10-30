from stable_baselines3.common.evaluation import evaluate_policy
from sb_training import make_env
import wandb
import random
import sys
from copy import deepcopy
import traceback
from tqdm import tqdm
import statistics
import numpy as np


class Random_Model:
	def __init__(self, action_space_size) -> None:
			self.action_space_size = action_space_size

	def predict(self,
			 	observations,  # type: ignore[arg-type]
				state=None,
				episode_start=None,
				deterministic=None):
			return (np.array([random.randint(0, self.action_space_size -1)]), None)


def eval_random_agent(config, base_worker_id):

	run = wandb.init(
			project="ReMEMber",
			config=config,
			sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
			monitor_gym=False,  # auto-upload the videos of agents playing the game
			save_code=False,  # optional
			#mode='disabled'        # this makes it so nothing is logged and useful to avoid logging debugging runs
	)

	model = Random_Model(action_space_size=5)
	env = make_env(config, base_worker_id)

	
	'''
	from stable_baselines3 import DQN
	dqn_model = DQN('CnnPolicy', env, verbose=2)
	obs = np.random.randint(0, 255, size=(1, 3, 84, 84), dtype=np.uint8)
	dqn_action = dqn_model.predict(obs)
	action = model.predict(obs)
	
	print('type(dqn_action):', type(dqn_action))
	print('dqn_action:', dqn_action)
	print('action:', action)
	exit()
	'''


	eval_ep_reward_means, eval_ep_lens = evaluate_policy(model, env, n_eval_episodes=5*24* config['eval_ep_batch_limit'], return_episode_rewards=True)

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


	wandb.log({'trial_completed': True})

	env.close()
	run.finish()


if __name__ == '__main__':

	tasks = [
		'RecipeRecall',
		'AllergicRobot',
		'MatchingPairs',
		'Hallway',
		'NighttimeNibble'
	]
		
	base_config = {
		"policy_type": "CnnPolicy",
		"total_timesteps": 500000,
		"eval_every_n_steps": 2083,
		"eval_ep_batch_limit": 10,
		"verbosity": 2,
		"os": "linux",
		"parallelism_override": "single_agent",
		"task_configs": {},
		"task_variant": "standard",
		"algo_name": "random",
		"trial_num": 0,
		"seed": 0,
		"parallelism": "single_agent"
	}	

	for task_num, task in tqdm(enumerate(tasks), desc='task'):
		config = deepcopy(base_config)
		config['env_name'] = task
		
		try:
			eval_random_agent(config=config, base_worker_id=task_num)
		except:
			print('\n\n||||||||||||||||||||||||| ERROR |||||||||||||||||||||||||\n')
			traceback.print_exc()
			print('\n||||||||||||||||||||||||| ERROR |||||||||||||||||||||||||\n\n')

	sys.exit('Training Complete.')
	exit()
	quit()