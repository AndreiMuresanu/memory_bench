from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper


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


if __name__ == '__main__':
	input_checking()