from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.envs.unity_aec_env import UnityAECEnv
from PIL import Image
import numpy as np


def input_checking():
	# unity_env = UnityEnvironment('/h/andrei/memory_bench/Builds/AllergicRobot/standard/linux/pixel_input/single_agent/gamefile.x86_64')
	unity_env = UnityEnvironment(
		file_name='/h/andrei/memory_bench/Builds/AllergicRobot/standard/linux/pixel_input/single_agent/gamefile.x86_64',
    	additional_args=["-logfile", "-"],
    	# additional_args=["-logfile", "-", "-screen-width", "100", "-screen-height", "100"],
	)
	env = UnityToGymWrapper(unity_env, uint8_visual=True)

	observation = env.reset()
	print(observation)
	print(type(observation))
	print('observation.shape:', observation.shape)
	for cur_step in range(30):
		action = env.action_space.sample()  # this is where you would insert your policy
		observation, reward, done, info = env.step(action)

		agent_view = Image.fromarray(observation)
		agent_view.save(f'./misc_results/agent_view_{cur_step}.png')
	
		if done: observation = env.reset()
	
	env.close()


def multi_agent_input_checking():
	# select the multi-agent executable
	unity_env = UnityEnvironment(
		file_name='/h/andrei/memory_bench/Builds/AllergicRobot/standard/linux/pixel_input/multi_agent/gamefile.x86_64',
    	additional_args=["-logfile", "-"],
	)
	env = UnityAECEnv(unity_env)

	env.reset()
	for iter_count, agent in enumerate(env.agent_iter(env.num_agents * 30)):
		step = iter_count // env.num_agents
		prev_observe, reward, done, info = env.last()
		action = 4  # 0: do nothing, 1: forward, 2: backward, 3: turn right, 4: turn left

		print(f'Agent: {iter_count % env.num_agents}, Step: {step}')
		agent_view = Image.fromarray((prev_observe['observation'][0] * 255).astype(np.uint8)).resize((200, 200))
		# agent_view.show() # displaying the agent view is very slow
		agent_view.save(f'./misc_results/agent-{iter_count % env.num_agents}_step-{step}.png')

		env.step(action)
	env.close()


if __name__ == '__main__':
	# input_checking()
	multi_agent_input_checking()