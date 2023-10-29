import os
import time


def cleaner():
	clean_every_n_mins = 60
	#clean_every_n_mins = 0.1
	dirs_to_clean = ['/h/andrei/memory_bench/training_scripts/wandb/', '/h/andrei/memory_bench/training_scripts/models/', '/h/andrei/memory_bench/training_scripts/runs/']

	while True:
		for dir in dirs_to_clean:
			#clean_cmd = f'du -sh {dir}'
			clean_cmd = f"find {dir} -name '*' -mmin +60 -delete"
			os.system(clean_cmd)
		print('cleaning cycle complete')
		time.sleep(60 * clean_every_n_mins)


if __name__ == '__main__':
	cleaner()
