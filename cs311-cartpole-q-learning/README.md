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
```

Where:

| Term | Meaning |
|---|---|
| Q(s, a) | Current Q-value for a state-action pair |
| alpha | Learning rate |
| reward | Reward received from the environment |
| gamma | Discount factor for future rewards |
| max(Q(next_state)) | Best expected future reward from the next state |

The program also uses **epsilon-greedy action selection**. Early in training, the agent explores more random actions. As training continues, epsilon decreases and the agent relies more on the Q-table.

## Environment

The project uses the Gymnasium `CartPole-v1` environment.

The observation space includes four values:

| Feature | Description |
|---|---|
| Cart position | Horizontal position of the cart |
| Cart velocity | Speed and direction of cart movement |
| Pole angle | Angle of the pole from vertical |
| Pole angular velocity | Speed and direction of pole rotation |

Because these values are continuous, the program discretizes them into bins before using them in the Q-table.

## Files in This Repository

```text
cartpole_q_learning_agent.py
cartpole_training_rewards.png
README.md
```

Optional report file:

```text
CS311_Assignment3_CartPole_RL_Report.docx
```

## Requirements

Install the required Python packages:

```powershell
python -m pip install "gymnasium[classic-control]" numpy matplotlib
```

If `python` does not work on your system, use:

```powershell
py -m pip install "gymnasium[classic-control]" numpy matplotlib
```

## How to Run

From the project folder, run:

```powershell
python .\cartpole_q_learning_agent.py
```

or:

```powershell
py .\cartpole_q_learning_agent.py
```

The program will:

1. Train the Q-learning agent.
2. Print training progress every 1,000 episodes.
3. Save a reward chart as `cartpole_training_rewards.png`.
4. Test the trained agent.
5. Print the test episode rewards.

## Example Results

One training run produced the following results:

```text
Best 100-episode training average: 328.05
Final 100-episode training average: 187.06
Average test reward: 430.50
Best test reward: 500.00
```

The test episode rewards were:

```text
Test Episode 1: 500.0
Test Episode 2: 230.0
Test Episode 3: 500.0
Test Episode 4: 500.0
Test Episode 5: 500.0
Test Episode 6: 495.0
Test Episode 7: 445.0
Test Episode 8: 237.0
Test Episode 9: 500.0
Test Episode 10: 398.0
```

The agent reached the maximum CartPole-v1 score of 500 in multiple test episodes.

## Training Reward Chart

The program saves a chart named:

```text
cartpole_training_rewards.png
```

This chart shows both the raw episode reward and the 100-episode moving average.

## Notes on Reinforcement Learning Results

Reinforcement learning results can vary between runs. This happens because the agent explores random actions, the environment starts from slightly different initial states, and the Q-table updates can follow different paths.

This project saves the best Q-table during training so the final test phase can use the strongest learned policy from the run.

## Limitations

This project uses tabular Q-learning, which requires the continuous CartPole observations to be converted into discrete bins. This makes the project easier to understand and implement, but it also loses some precision.

A future improvement would be to use a Deep Q-Network, which could learn from the continuous observation values directly instead of using manual discretization.

## References

Farama Foundation. (n.d.). *Cart Pole*. Gymnasium Documentation. https://gymnasium.farama.org/environments/classic_control/cart_pole/

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning, 8*, 279-292.
