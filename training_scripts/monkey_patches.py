
from mlagents_envs.envs.unity_pettingzoo_base_env import UnityPettingzooBaseEnv

#'''
# monkey patching to avoid unexpected argument 'seed' error in UnityParallelEnv.reset() 
def monkey_UnityParallelEnv_reset(self, seed=None, options=None):
	"""
	Resets the environment.
	"""
	UnityPettingzooBaseEnv.reset(self)
	return self._observations, {}
#'''


def monkey_VecEnvWrapper_get_attr(self, attr_name: str, indices = None):
	if attr_name == 'render_mode':
		raise AttributeError('This is causing issues so we are going to explicitly ignore it.')

	if indices is not None:
		raise Exception('Did not account for this.')

	return self.venv.get_attr(attr_name)

#'''
def monkey_SS_concat_obs(self, obs_dict):
	if self.black_death:
		self.obs_buffer[:] = 0
	for i, agent in enumerate(self.par_env.possible_agents):
		if self.black_death:
			if agent in obs_dict:
				# ORIGINAL:
				#self.obs_buffer[i] = obs_dict[agent]
				# NEW:
				assert len(obs_dict[agent]['observation']) == 1, 'I don\'t know how this should be handled if there is more then 1 observation'
				self.obs_buffer[i] = obs_dict[agent]['observation'][0]
		else:
			if agent not in obs_dict:
				raise AssertionError("environment has agent death. Not allowed for pettingzoo_env_to_vec_env_v1 unless black_death is True")
			self.obs_buffer[i] = obs_dict[agent]

	return self.obs_buffer.copy()
#'''
