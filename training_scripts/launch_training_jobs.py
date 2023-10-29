from copy import deepcopy
from sb_training import get_default_episode_length
import json
import os
from datetime import datetime
import time
import random
import traceback

SBATCH_CONFIG_FOLDER = '/h/andrei/memory_bench/training_scripts/sbatch_temp_configs'
SBATCH_SH_FOLDER = '/h/andrei/memory_bench/training_scripts/sbatch_temp_sh_scripts'
SBTACH_BASEFILE_PATH = '/h/andrei/memory_bench/training_scripts/sb_training_sbatch_basefile.sh'
TRAINING_CMD = 'xvfb-run -s "-screen 0 100x100x24" -a python sb_training.py'


def get_config_path():
	current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
	return os.path.join(SBATCH_CONFIG_FOLDER, f'{current_time}_config.json'), current_time

def create_config_file(config):
	# make sure that the config doesnt already exist
	
	config_path, unique_time = get_config_path()

	# find non-existing config path	
	while os.path.exists(config_path):
		time.sleep(random.uniform(0, 1))
		config_path, unique_time = get_config_path()

	with open(config_path, 'w') as f:
		json.dump(config, f, indent=4)

	# this is a step to verify that the contents on the config file are correct
	# it is unlikely but possible that a different sbatch scipt got its config path at the same time and overwrote our time
	with open(config_path, 'r') as f:
		verification_config = json.load(f)

	if verification_config != config:
		# leave the config file alone so the other job can run it's job
		# and create a new config file
		return create_config_file()
	else:
		return config_path, unique_time


def create_sbatch_file(config_path, unique_time):
	with open(SBTACH_BASEFILE_PATH, 'r') as f:
		sbatch_basefile_txt = f.read()

	sbatch_basefile_txt += f'\n{TRAINING_CMD} "{config_path}"'

	sbatch_path = os.path.join(SBATCH_SH_FOLDER, f'{unique_time}_job-cmd.sh')
	with open(sbatch_path, 'w') as f:
		f.write(sbatch_basefile_txt)

	return sbatch_path


def launch_sbatch_job(config):
	config_path, unique_time = create_config_file(config)
	sbatch_filepath = create_sbatch_file(config_path, unique_time)
	os.system(f'sbatch {sbatch_filepath}')


def main():
	task_names = [
		#('RecipeRecall', {}),
		#('AllergicRobot', {}),
		#('MatchingPairs', {}),
		('Hallway', {}),
		#('NighttimeNibble', {}),
		
		# ('AllergicRobot', {
		#  	'episode_step_count': 100,
		#  	'max_ingredient_count': 10,
		#  	'available_ingredient_count': 1,
		#  	'allergic_prob': 0.5,
		#  	'allergic_tastiness': -5,
		#  	'love_prob': 0.5,
		#  	'love_tastiness': 5
		# }),
		# ('AllergicRobot', {
		# 	'episode_step_count': 100,
		# 	'max_ingredient_count': 30,
		# 	'available_ingredient_count': 1,
		# 	'allergic_prob': 0.5,
		# 	'allergic_tastiness': -5,
		# 	'love_prob': 0.5,
		# 	'love_tastiness': 5
		# }),
		#('MatchingPairs', {
		#	'max_ingredient_count': 20,
		#	'available_ingredient_count': 10
		#})
	]

	task_variants = {
		'standard',
		#'partial_observability',
	}

	# can't store the algos directly because we want to be able to directly upload the config dict to wandb
	algo_names = [
		'RecurrentPPO',
		#'PPO',
		#'A2C',
		#'DQN',
	]

	base_config = {
		"policy_type": "CnnPolicy",
		#"total_timesteps": 1_000_000,
		"total_timesteps": 500_000,
		#"total_timesteps": 1000,
		#"eval_every_n_steps": 4166,	#this is total_timesteps_so_far / 24
		"eval_every_n_steps": 2083,	#this is total_timesteps_so_far / 24
		#"eval_every_n_steps": 42,	#this is total_timesteps_so_far / 24
		"eval_ep_batch_limit": 10,
		#"verbosity": 0,
		"verbosity": 2,
		"os": 'linux',
		"parallelism_override": "single_agent",
	}

	#which_trials = [0, 1, 2, 4]
	which_trials = [0, 3, 4]

	# num_of_trial_repeats = 3
	# #num_of_trial_repeats = 1

	# trial_offset = 0	# this is the number of runs already completed, this will also set the seed
	# #trial_offset = 4	# this is the number of runs already completed, this will also set the seed
	
	# base_config['num_of_trial_repeats'] = num_of_trial_repeats
	# base_config['trial_offset'] = trial_offset


	cur_worker_id = 0
	for task_settings in task_names:
		for task_variant in task_variants:
			for algo_name in algo_names:
				#for trial_num in range(num_of_trial_repeats):
				for trial_num in which_trials:
					config = deepcopy(base_config)	# I think deepcopy is likely not needed
					config['env_name'] = task_settings[0]
					config['task_configs'] = task_settings[1]

					config['task_variant'] = task_variant
					config['algo_name'] = algo_name

					# config['trial_num'] = trial_num + trial_offset
					# config['seed'] = trial_num + trial_offset
					config['trial_num'] = trial_num
					config['seed'] = trial_num
		
					if 'parallelism_override' in config:
						config['parallelism'] = config['parallelism_override']
					else:
						if config['env_name'] == 'Hallway':
							config['parallelism'] = 'single_agent'
						else:
							config['parallelism'] = 'multi_agent'
					
					if config['parallelism'] == 'multi_agent' and 'episode_step_count' not in config['task_configs']:
						config['task_configs']['episode_step_count'] = get_default_episode_length(config['env_name'])
					
					# convert all this sloppy code into a factory
					if algo_name == 'RecurrentPPO':
						config['policy_type'] = 'CnnLstmPolicy'
					
					try:
						launch_sbatch_job(config)
						#multi_agent_input_checking(config)
					except:
						print('\n\n||||||||||||||||||||||||| ERROR |||||||||||||||||||||||||\n')
						traceback.print_exc()
						print('\n||||||||||||||||||||||||| ERROR |||||||||||||||||||||||||\n\n')

					cur_worker_id += 2

	print('\nDONE')



if __name__ == '__main__':
	main()