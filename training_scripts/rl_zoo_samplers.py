from typing import Any, Dict

import numpy as np
import optuna
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn

from rl_zoo3 import linear_schedule


'''
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
#'''


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


def sample_ppo_params(trial: optuna.Trial, parallelism) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    if parallelism == 'single_agent':
        if batch_size > n_steps:
            batch_size = n_steps
    elif parallelism == 'multi_agent':
        if batch_size > n_steps * 24:
            batch_size = n_steps * 24
    else:
        raise ValueError(f'Parallelization is invalide: {parallelization}')

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    # Toggle PyTorch RMS Prop (different from TF one, cf doc)
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # sde_net_arch = trial.suggest_categorical("sde_net_arch", [None, "tiny", "small"])
    # full_std = trial.suggest_categorical("full_std", [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch]

    # sde_net_arch = {
    #     None: None,
    #     "tiny": [64],
    #     "small": [64, 64],
    # }[sde_net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "normalize_advantage": normalize_advantage,
        "max_grad_norm": max_grad_norm,
        "use_rms_prop": use_rms_prop,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            # full_std=full_std,
            activation_fn=activation_fn,
            # sde_net_arch=sde_net_arch,
            ortho_init=ortho_init,
        ),
    }



def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)

    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])

    net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    #if trial.using_her_replay_buffer:
    #    hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams