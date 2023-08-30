from mlagents_envs.environment import UnityEnvironment
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn
import sys
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import optuna
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from PIL import Image
import os

import gymnasium as gym
import numpy as np
#import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
#from wandb.integration.sb3 import WandbCallback
from optuna.integration.wandb import WeightsAndBiasesCallback

from copy import deepcopy
from tqdm import tqdm

from sb_training import make_env


PROJECT_DIR_PATH = '/h/andrei/memory_bench'

N_EVALUATIONS = 50
N_TIMESTEPS = int(250000)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
# convert all this sloppy code into a factory
print(sys.argv)
task_name = sys.argv[1]
print("TASK NAME................", task_name)

algo_name = sys.argv[2]
print("ALGO NAME................", algo_name)

worker_id = int(sys.argv[3])
print("worker_id................", worker_id)

task_names = [
	'AllergicRobot',
	'MatchingPairs',
	'Hallway',
	'RecipeRecall'
]

# can't store the algos directly because we want to be able to directly upload the config dict to wandb
algo_names = [
	'PPO',
	'RecurrentPPO',
	'A2C',
	'DQN',
	'SAC',
]
assert task_name in task_names
assert algo_name in algo_names
wandb_kwargs = {
        "project":"ReMEMber",
        "entity": "team-andrei",
    "sync_tensorboard":True,  # auto-upload sb3's tensorboard metrics
    "monitor_gym":True,  # auto-upload the videos of agents playing the game
    "save_code":True,  # optional
    "name": "{}_{}_250ksteps".format(task_name, algo_name)
    }
wandbc = WeightsAndBiasesCallback(metric_name="mean reward", wandb_kwargs=wandb_kwargs, as_multirun=True)

base_config = {
	"policy_type": "CnnPolicy",
	"total_timesteps": N_TIMESTEPS,
    }
	#"total_timesteps": 3,

def sample_a2c_params(trial: optuna.Trial):
    """Sampler for A2C hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = [
        {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    ]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        },
    }

def sample_dqn_params(trial: optuna.Trial): # -> dict[str, Any]:
    """Sampler for recurrent PPO hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    tau = trial.suggest_float("tau", 0, 1)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    train_freq = 2 ** trial.suggest_int("train_freq", 1, 5)
    batch_size = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    buffer_size = 10 ** trial.suggest_int("buffer_size", 3, 7)
    learning_starts = 5 * 10 ** trial.suggest_int("buffer_size", 1, 4)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)

    # Display true values.
    trial.set_user_attr("tau", tau)
    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("train_freq", train_freq)
    trial.set_user_attr("buffer_size", buffer_size)
    trial.set_user_attr("gamma", gamma)
    trial.set_user_attr("max_grad_norm", max_grad_norm)
    trial.set_user_attr("learning_starts", learning_starts)
    trial.set_user_attr("learning_rate", learning_rate)


    return {
        "tau": tau,
        "batch_size": batch_size,
        "gamma": gamma,
        "train_freq": train_freq,
        "buffer_size": buffer_size,
        "max_grad_norm": max_grad_norm,
        'learning_starts': learning_starts,
        'learning_rate': learning_rate
    }

def sample_rppo_params(trial: optuna.Trial): # -> dict[str, Any]:
    """Sampler for recurrent PPO hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 7, 12)
    batch_size = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)
    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("gamma", gamma)
    trial.set_user_attr("max_grad_norm", max_grad_norm)


    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            wandb.log({"mean_reward": self.last_mean_reward, "eval_idx": self.eval_idx})
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

#def sb_training(config):
@wandbc.track_in_wandb()
def objective(trial: optuna.Trial) -> float:

    config = base_config.copy()

    config['env_name'] = task_name
    config['algo_name'] = algo_name
    #config['trail_num'] = trail_num
    if algo_name == 'RecurrentPPO':
        config['policy_type'] = 'CnnLstmPolicy'
    print("config", config)
    env = make_env(config)
    kwargs = {"policy": config['policy_type'],
              "env": env,
              "verbose": 2,
              #"tensorboard_log": f"runs/{run.id}"
              }
    #if trial._trial_id == 0:
    #    kwargs.update(trial.system_attrs['fixed_params'])
    #else:
    if "PPO" in algo_name:
        kwargs.update(sample_rppo_params(trial))
    elif "A2C" in algo_name:
        kwargs.update(sample_a2c_params(trial))
    elif "DQN" in algo_name or "SAC" in algo_name:
        kwargs.update(sample_dqn_params(trial))
        if "SAC" in algo_name:
            del kwargs['max_grad_norm']
    print("kwargs", kwargs)

    #eval_callback = WandbCallback(gradient_save_freq=100,model_save_path=f"models/{run.id}",verbose=2)
    eval_callback = TrialEvalCallback(
        env, trial, n_eval_episodes=3, eval_freq=EVAL_FREQ, deterministic=True
    )

    algo = get_algo(config['algo_name'])
    model = algo(**kwargs)
    nan_encountered = False
    try:
        model.learn(
        	total_timesteps=config["total_timesteps"],
        	callback=eval_callback,
        )
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        #model.env.close()
        env.close()
        #run.finish()
    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


def get_env_path(task_name):
	relative_env_paths = {
		'AllergicRobot': 'Builds/AllergicRobot/standard/linux/pixel_input/single_agent/gamefile.x86_64',
		'MatchingPairs': 'Builds/MatchingPairs/standard/linux/pixel_input/single_agent/gamefile.x86_64',
        'Hallway': 'Builds/Hallway/standard/linux/pixel_input/single_agent/gamefile.x86_64',
        'RecipeRecall': 'Builds/RecipeRecall/standard/linux/pixel_input/single_agent/gamefile.x86_64',
	}
	return os.path.join(PROJECT_DIR_PATH, relative_env_paths[task_name])


def get_algo(alg):
	algos = {
		'PPO': PPO,
		'RecurrentPPO': RecurrentPPO,
		'SAC': SAC,
		'A2C': A2C,
		'DQN': DQN,
	}
	return algos[alg]


if __name__ == '__main__':
        # Set pytorch num threads to 1 for faster training.

    N_TRIALS = 500
    torch.set_num_threads(1)

    #sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    sampler = TPESampler()
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    try:
        #study.optimize(objective, n_trials=N_TRIALS, timeout=600,callbacks=[wandbc])
        study.optimize(objective, n_trials=N_TRIALS, timeout=None,callbacks=[wandbc])
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))