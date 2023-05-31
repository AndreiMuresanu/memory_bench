from mlagents_envs.environment import UnityEnvironment


def main():
	env_location = 'C:\Users\Andrei\Documents\vector_institute\memory_palace_stuff\unity_projects\memory_palace_2\Builds'
	env = UnityEnvironment(file_name=env_location)


if __name__ == '__main__':
	main()