from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import pygame
import numpy as np
import cv2
import time
from datetime import datetime
import os
import csv


class human_env():
	def __init__(self, env_name, path_to_env, save_dir, player_name) -> None:
		self.fps = 8
		self.score_screen_duration = 2	#in seconds
		self.window_size = 8 * 84  #the size of the PyGame window
		self.env_name = env_name
		self.save_dir = save_dir
		self.player_name = player_name

		unity_env = UnityEnvironment(path_to_env, seed=0)
		self.env = UnityToGymWrapper(unity_env, uint8_visual=True)

		pygame.init()
		self.window = pygame.display.set_mode((self.window_size, self.window_size))
		self.clock = pygame.time.Clock()
		
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


	def play_game(self, episode_count=99999):
		filename = f"{self.player_name}_{self.env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
		with open(os.path.join(self.save_dir, filename), 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(['cur_episode', 'cumulative_reward'])
		
		observation = self.env.reset()
		playing_game = True
		cumulative_reward = []
		episode_reward = 0
		cur_episode = 0
		while playing_game:
			self.clock.tick(self.fps)

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
				
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					playing_game = False

			observation, reward, done, info = self.env.step(action) #lol, this is different from both ml-agents docs and gymnasium docs

			episode_reward += reward
			cumulative_reward.append(episode_reward)

			self.render_frame(observation)
			self.render_info(episode_reward)
			pygame.display.update()
			
			if done:
				print('finished episode:', cur_episode, 'final score:', episode_reward)
				with open(os.path.join(self.save_dir, filename), 'a', newline='') as file:
					writer = csv.writer(file)
					writer.writerow([cur_episode] + cumulative_reward)

				self.display_final_score(episode_reward)
				pygame.display.update()
				time.sleep(self.score_screen_duration)

				observation = self.env.reset()
				cumulative_reward = []
				episode_reward = 0
				cur_episode += 1

				if cur_episode >= episode_count:
					break
		self.env.close()


if __name__ == '__main__':
	save_dir = r'./human_results'
	#player_name = 'andrei'
	#player_name = 'george'
	#player_name = 'tati'
	# player_name = 'darci'
	player_name = 'test'

	#env_info = ('AllergicAgent', 32, r'../unity_projects\memory_palace_2\Builds\AllergicAgent\windows\pixel_input\single_agent\memory_palace_2.exe')
	#env_info = ('Hallway', 5, r'../unity_projects\memory_palace_2\Builds\Hallway\windows\pixel_input\single_agent\memory_palace_2.exe')
	# env_info = ('MatchingPairs', 5, r'../unity_projects\memory_palace_2\Builds\MatchingPairs\medium\windows\pixel_input\single_agent\memory_palace_2.exe')
	env_info = ('MatchingPairs', 5, r'/h/andrei/memory_bench/Builds/AllergicRobot/standard/linux/pixel_input/single_agent/gamefile.x86_64')
	#env_info = ('RecipeRecall', 27, r'../unity_projects\memory_palace_2\Builds\RecipeRecall\windows\pixel_input\single_agent\memory_palace_2.exe')
	
	env = human_env(env_name=env_info[0], path_to_env=env_info[2], save_dir=save_dir, player_name=player_name)
	env.play_game(episode_count=env_info[1])