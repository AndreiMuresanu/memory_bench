# ReMEMber: Assessing Memory Capabilities of RL Agents
ArXiv | [Colab](https://colab.research.google.com/drive/1E3TPO9z1cOH1p63lvwxeWon40xn5Ekvc?usp=sharing)
Benchmarking RL memory

# Table of Contents

# Abstract
Memory is a critical component required for intelligent decision-making and remains an elusive frontier in the realm of reinforcement learning (RL) agents. Surprisingly, no standardized method exists to gauge the memory capabilities of RL agents, stalling progress in developing truly intelligent systems. Addressing this gap, we present ReMEMber —a benchmark that improves the assessment of RL agents’ memory capabilities. Drawing inspiration from studies in psychology, ReMEMber introduces an evaluation framework that evaluates multiple different aspects of memory, including spatial, associative, episodic, and implicit, among others. Spanning across four distinctive tasks, offering increasing levels of difficulty, and using three different input modalities, ReMEMber enables the evaluation of an RL agents’ memory abilities. We benchmark these datasets on popular RL algorithms, including PPO and SAC. By leveraging this benchmark, researchers and practitioners can gain valuable insights into RL agents’ memory capacity and performance, facilitating advancements in memory-enhanced RL algorithms and fostering the development of more intelligent decision-making systems.

# Memory Tasks

## Allergic Robot
- screenshots
- videos

The allergic robot task is designed to evaluate an agent's ability to recall strong emotional events akin to flashbulb memories (a type of episodic memory). In this task, there are a total of 48 unique foods (the number of unique foods is modifiable). Each round, 40 random foods are spawned in the environment (the number of spwaned food is modifiable). At the beginning of the task, each food is assigned a value that determines the associated reward according to three categories: foods the agent is allergic to, foods that it loves, and normal food. The allergy-inducing foods have a 10\% probability of occurrence and are associated with a reward of -5, representing a strong negative reaction for the agent (this reward and probability are modifiable). The tasty foods also have a 10\% probability of occurrence and are associated with a reward of +5, representing a highly positive reaction for the agent (this reward and probability are modifiable). The remaining 80\% of the foods are considered normal, and their rewards are uniformly sampled in [-0.1, 1] (this reward and probability are modifiable).

The objective of the task is for the agent to to identify and remember which specific food items it is allergic to in order to avoid incurring the negative emotional consequence. At the same time, it should identify the foods it loves. By incorporating emotional rewards and consequences, the task elicits the agent's ability to recall and utilize information related to strong emotional events.

## Recipe Recall
- screenshots
- videos

Recipe recall involves presenting an agent with a large set of randomly selected ingredients. There are a total of 16 food types and 50 unique recipes in the environment (these are default values which are modifiable). Each recipe is assigned a tastiness value ranging from -0.5 to +5, representing its tastiness. The inclusion of ingredients in a recipe is randomized, with each ingredient having a 15\% chance of being included.

During each episode, the agent is provided with 6 random ingredients to work with (the number of ingredients is modifiable). The objective is for the agent to identify and select the ingredients corresponding to a specific recipe. If the agent successfully touches the ingredients that match a recipe, it is rewarded with the tastiness value associated with that recipe. However, if the agent fails to select the correct ingredients, it receives no reward for that  episode. In order to achieve a high reward, the agent must demonstrate the ability to remember multiple recipes, potentially spanning across different episodes or sessions. This task serves as a means to test the agent's factual (semantic) memory as it learns what recipes are, specifically its capacity to retain and recall discrete facts accurately. 

## Matching Pairs
- screenshots
- videos

In the matching pairs task, there are two pairs of objects scattered randomly around a room (the number of pairs of objects is modifiable). Each pair of objects has the same appearance, but their positions are random. The goal of the agent is to touch one object from a pair and then touch its corresponding pair without touching any other objects in between, and then repeat that for the other pair.

If the agent successfully touches a pair of objects without any intervening touches, it receives a reward of +5 (maximum reward of +10 for both pairs). However, if the agent fails to touch any pairs or touches other objects in between, it receives a reward of 0. All of these rewards are modifiable.

The task tests an agent's ability to make associations (episodic memory) and learn the relationship between seemingly unrelated items. Upon touching the first block, the agent needs to encode and store its identity in memory. It must then explore the room while remembering the initial block it touched until it can associate the block to its matching pair. We also implement this task in a partially-observable mode, where the agent has a far more restricted field of view. This makes the task harder, as the  agent is forced to retain an object in its memory until it sees a pair, whereas the task is easier if the agent has a larger field of view and can see multiple objects at the same time. This might be similar to a situation where a robot in a house must learn to make associations between objects and interact with them in an implicit order.

## Hallway
- screenshots
- videos

In this task, the agent is initially presented with a goal token, represented by a yellow O or a blue X, concealed behind a wall. The agent's objective is to traverse down a hallway, no longer seeing the goal token, and touch a door that corresponds to the matching token. The location of the goal token is randomized across episodes, which tests the agent's episodic memory, as it needs to remember the explicit symbols observed during its lifetime. The agent must recall and track the changing location of the goal token. The agent receives a negative reward of -0.1 for each step taken in the environment to minimize unnecessary movements. Additionally, a penalty of -0.1 is imposed if the agent touches the door with the incorrect token. If the agent successfully touches the correct target, it receives a reward of +5.

## Robotics Modality
- screenshots
- videos
- description

## Vector Modality
- screenshots
- videos
- description

## Partial Observability
- screenshots
- videos
- description

## Multi-Task
- screenshots
- videos
- description

# Setup
- should be extremely simple (follow colab code)
- link mlagents install tutorials

# Quickstart

## Colab Examples
### Gym API
### PettingZoo API
### Unity API
### Human Control
- video of human control on allergic agent
- path to human control script
- short description

## Virtual Display

# Additional Documentation
- link useful docs

# Comments
- cite relevant code sources

# BibTex
[awaiting ArXiv release]
