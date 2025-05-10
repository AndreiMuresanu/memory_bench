Configuration Parameters
===

# General Environment Parameters

## Usage

```python
from mlagents_envs.environment import UnityEnvironment

unity_env = UnityEnvironment('/content/memory_bench/unity_projects/memory_palace_2/Builds/AllergicAgent/linux/pixel_input/multi_agent/gamefile.x86_64', # the path to the task executable
                              seed=2 # sets the seed for all random processes within the task environment
                            )
```

## Parameter Options

Taken from the ML-Agents Low-Level Python API docs (https://unity-technologies.github.io/ml-agents/Python-LLAPI/#loading-a-unity-environment). See the docs for more details.

- file_name: is the name of the environment binary (located in the root directory of the python project).
- worker_id: indicates which port to use for communication with the environment. For use in parallel training regimes such as A3C.
- seed: indicates the seed to use when generating random numbers during the training process. In environments which are stochastic, setting the seed enables reproducible experimentation by ensuring that the environment and trainers utilize the same random seed. IMPORTANT: This seed does not control environment initialization. For example, in Allergic Robot this seed will NOT control the tastiness of each food. To control initialization stochasticity you must use the "initialization_seed".
- side_channels: provides a way to exchange data with the Unity simulation that is not related to the reinforcement learning loop. For example: configurations or properties. More on them in the Side Channels doc (https://unity-technologies.github.io/ml-agents/Custom-SideChannels/).


# General Configuration Parameters

## Usage

```python
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel import EngineConfigurationChannel

config_channel = EngineConfigurationChannel()
config_channel.set_configuration_parameters(time_scale=100.0) # The time_scale parameter defines how quickly time will pass within the simulation

unity_env = UnityEnvironment('/content/memory_bench/unity_projects/memory_palace_2/Builds/AllergicAgent/linux/pixel_input/multi_agent/gamefile.x86_64',
                              side_channels=[config_channel])
```

## Parameter Options

Taken from the ML-Agents Low-Level Python API docs (https://unity-technologies.github.io/ml-agents/Python-LLAPI/#communicating-additional-information-with-the-environment). See the docs for more details.

- width: Defines the width of the display. (Must be set alongside height)
- height: Defines the height of the display. (Must be set alongside width)
- quality_level: Defines the quality level of the simulation.
- time_scale: Defines the multiplier for the deltatime in the simulation. If set to a higher value, time will pass faster in the simulation but the physics may perform unpredictably.
- target_frame_rate: Instructs simulation to try to render at a specified frame rate.
- capture_frame_rate Instructs the simulation to consider time between updates to always be constant, regardless of the actual frame rate.


# Task Parameter Usage Example

```python
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel import EnvironmentParametersChannel

setup_channel = EnvironmentParametersChannel()
setup_channel.set_float_parameter("max_ingredient_count", -1)

unity_env = UnityEnvironment('/content/memory_bench/unity_projects/memory_palace_2/Builds/AllergicAgent/linux/pixel_input/multi_agent/gamefile.x86_64',
                              side_channels=[setup_channel])
```

# Allergic Robot

- (int) max_ingredient_count:  The size of the pool of ingredients to draw from each round. Can set to -1 to allow for the largest possible food count.
- (int) available_ingredient_count: The number of ingredients to sample each round.
- (float) allergic_prob: The probability that that the agent is allergic to a particular ingredient (0.1 == 10% probability).
- (float) allergic_tastiness: The tastiness of an allergic ingredient (ex. -5)
- (float) love_prob: The probability that that the agent loves a particular ingredient (0.1 == 10% probability).
- (float) love_tastiness: The tastiness of an ingredient the agent loves (ex. 5)
- (float) min_normal_tastiness: The minimum tastiness of a normal food (ex. -0.1)
- (float) max_normal_tastiness: The maximum tastiness of a normal food (ex. 1)

Note that the probability a food is normal is: 1 - (allergic_prob + love_prob)

## (Optional) Custom SideChannel For Specifying Exact Rewards

Parameter side channels only support float parameters. To specify a desired list of food rewards, you may use the ListSideChannel class. Specifying this parameter will override max_ingredient_count, allergic_prob, allergic_tastiness, love_prob, love_tastiness, min_normal_tastiness, max_normal_tastiness. Example usage:

TODO: location of this import may change
```python
from custom_side_channels import ListSideChannel


reward_channel = ListSideChannel()
reward_channel.send_list([1.0, 2.5, 3.0])  # this is the list of rewards you want your foods to have

unity_env = UnityEnvironment('/content/memory_bench/unity_projects/memory_palace_2/Builds/AllergicAgent/linux/pixel_input/multi_agent/gamefile.x86_64',
                              side_channels=[reward_channel])
```

# Recipe Recall

- (int) max_ingredient_count:  The size of the pool of ingredients to draw from each round. Can set to -1 to allow for the largest possible food count.
- (int) available_ingredient_count: The number of ingredients to sample each round.
- (int) recipe_count: The total number of recipes in the recipe book.
- (uint) recipe_prep_time: The number of frames available to the agent to select their ingredients.
- (float) ingredient_include_prob: The probability a particular ingredient is included within any one recipe.
- (float) min_tastiness: The minimum tastiness of a recipe.
- (float) max_tastiness: The maximum tastiness of a recipe.

# Matching Pairs

- (int) max_ingredient_count:  The size of the pool of ingredients to draw from each round. Can set to -1 to allow for the largest possible food count.
- (int) available_ingredient_count: The number of ingredients to sample each round.
- (float) match_reward: The amount of reward recieved each time the agent matches a pair.

# Hallway

None

# custom modalities ??

- make camera clipping range a parameters so that all tasks can have a partial observability parameter
