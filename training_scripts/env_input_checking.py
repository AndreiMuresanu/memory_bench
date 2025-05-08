import gymnasium as gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from PIL import Image


def main():
   #path_to_env = r'C:\Users\Andrei\Documents\vector_institute\memory_palace_stuff\unity_projects\memory_palace_2\Builds\windows\pixel_input'
   # path_to_env = '/h/andrei/memory_bench/Builds/TempLinux/Hallway/gamefile.x86_64'
   # path_to_env = '/h/andrei/memory_bench/Builds/AllergicRobot/standard/linux/pixel_input/single_agent/gamefile.x86_64'
   path_to_env = f'/h/andrei/memory_bench/Builds/MatchingPairs/standard/linux/pixel_input/single_agent/gamefile.x86_64'
   unity_env = UnityEnvironment(path_to_env)

   print('========== created Unity Env ==========')

   env = UnityToGymWrapper(unity_env, uint8_visual=True)
   #env = gym.make("LunarLander-v2", render_mode="human")
   
   print('========== created Gym Env ==========')

   observation = env.reset()
   for cur_step in range(5):
      print('cur_step:', cur_step)

      #action = env.action_space.sample()  # this is where you would insert your policy
      action = 1

      observation, reward, done, info = env.step(action) #lol, this is different from both ml-agents docs and gymnasium docs

      print('observation:', observation)
      print('type(observation):', type(observation))
      print('observation.shape:', observation.shape)
      print('action:', action)
      print('reward:', reward)
      print('done:', done)
      print('info:', info)

      if cur_step % 1 == 0:
         agent_view = Image.fromarray(observation)
         #agent_view.show()
         agent_view.save(f'./misc_results/agent_view_{cur_step}.png')

      if done:
         observation = env.reset()
   env.close()


if __name__ == '__main__':
   main()