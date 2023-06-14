import gymnasium as gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from PIL import Image
import pygame
from inputimeout import inputimeout
import numpy as np
import cv2
import time


class human_env():
	def __init__(self, path_to_env) -> None:
		self.user_time_per_frame = 2	#in seconds
		self.score_screen_duration = 2	#in seconds
		self.window_size = 8 * 84  # The size of the PyGame window

		unity_env = UnityEnvironment(path_to_env, seed=2)
		self.env = UnityToGymWrapper(unity_env, uint8_visual=True)

		pygame.init()
		self.window = pygame.display.set_mode((self.window_size, self.window_size))
		
		self.info_font = pygame.font.SysFont("Arial", 28, bold=True)
		self.final_score_font = pygame.font.SysFont("Arial", 70, bold=True)


	def render_frame(self, observation):
		surf = pygame.surfarray.make_surface(cv2.resize(np.rot90(observation), dsize=(self.window_size, self.window_size)))
		self.window.blit(surf, (0, 0))

	def render_info(self, episode_reward):
		score_dis = self.info_font.render(f'Score: {round(episode_reward, 2)}', 1, (0,0,0))
		self.window.blit(score_dis, (0, 0))

	def display_final_score(self, episode_reward):
		self.window.fill((90, 170, 149))
		score_dis = self.final_score_font.render(f'Final Score: {round(episode_reward, 2)}', 1, (0,0,0))
		self.window.blit(score_dis, (50, int(self.window_size / 2) - 50 ))


	def play_game(self):
		observation = self.env.reset()
		playing_game = True
		#for cur_step in range(100):
		episode_reward = 0
		while playing_game:
			user_inp = pygame.key.get_pressed()
			if user_inp[pygame.K_w] or user_inp[pygame.K_UP]:
				action = 1
			elif user_inp[pygame.K_s] or user_inp[pygame.K_DOWN]:
				action = 2
			elif user_inp[pygame.K_a] or user_inp[pygame.K_LEFT]:
				action = 3
			elif user_inp[pygame.K_d] or user_inp[pygame.K_RIGHT]:
				action = 4
			else:
				action = 0
				
			observation, reward, done, info = self.env.step(action) #lol, this is different from both ml-agents docs and gymnasium docs

			episode_reward += reward

			#this resets the env and starts a new epoch
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					#done = True
					playing_game = False

			self.render_frame(observation)
			self.render_info(episode_reward)
			pygame.display.update()
			
			if done:
				self.display_final_score(episode_reward)
				pygame.display.update()
				time.sleep(self.score_screen_duration)

				observation = self.env.reset()
				episode_reward = 0
		self.env.close()


if __name__ == '__main__':
	#path_to_env = r'C:\Users\georg\Documents\andreis_shit\memory_bench\unity_projects\memory_palace_2\Builds\Hallway\windows\pixel_input\memory_palace_2.exe'
	#path_to_env = r'C:\Users\georg\Documents\andreis_shit\memory_bench\unity_projects\memory_palace_2\Builds\RecipeRecall\windows\pixel_input\memory_palace_2.exe'
	path_to_env = r'C:\Users\georg\Documents\andreis_shit\memory_bench\unity_projects\memory_palace_2\Builds\AllergicAgent\windows\pixel_input\single_agent\memory_palace_2.exe'
	
	env = human_env(path_to_env=path_to_env)
	env.play_game()