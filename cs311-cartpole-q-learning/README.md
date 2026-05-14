# CS311 CartPole Q-Learning Agent

## Project Overview

This project was created for CS311 Assignment 3: Developing a Reinforcement Learning Agent.

The goal of the project is to train a reinforcement learning agent to balance a pole on a moving cart using the `CartPole-v1` environment from Gymnasium. The agent learns through trial and error by interacting with the environment, receiving rewards, and updating a Q-table.

The agent uses Q-learning with an epsilon-greedy exploration strategy. Since CartPole observations are continuous, the program converts the observation values into discrete bins so they can be used in a Q-table.

## Agent Purpose

The CartPole Q-Learning Agent learns how to keep a pole balanced by choosing one of two actions:

| Action | Meaning |
|---|---|
| 0 | Push cart left |
| 1 | Push cart right |

The agent receives a reward for each time step that the pole stays balanced. Over many training episodes, the agent learns which actions are better in different states.

## Algorithm Used

This project uses **Q-learning**, a reinforcement learning algorithm that learns action values for different states.

The Q-learning update rule is:

```text
Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(next_state)) - Q(s, a)]
