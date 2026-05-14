"""
CS311 Assignment 3
Developing a Reinforcement Learning Agent

Project:
CartPole Q-Learning Agent

Purpose:
This program trains a reinforcement learning agent to balance a pole
on a moving cart using the CartPole-v1 environment from Gymnasium.

The agent uses Q-learning. Since CartPole observations are continuous,
the program converts those observations into discrete bins so the values
can be stored in a Q-table.
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# Function: create_bins
# Purpose:
# Create bins that convert CartPole's continuous observation values
# into discrete state values for the Q-table.
# ------------------------------------------------------------
def create_bins():
    """
    Create discretization bins for the four CartPole observations.

    CartPole observations:
    1. Cart position
    2. Cart velocity
    3. Pole angle
    4. Pole angular velocity

    Returns:
        A list of NumPy arrays, one for each observation feature.
    """

    # Cart position matters, but not as much as pole angle.
    cart_position_bins = np.linspace(-2.4, 2.4, 10)

    # Cart velocity can vary widely.
    cart_velocity_bins = np.linspace(-4.0, 4.0, 10)

    # Pole angle is very important, so we use more bins here.
    pole_angle_bins = np.linspace(-0.2095, 0.2095, 18)

    # Pole angular velocity is also very important, so we use more bins here.
    pole_angular_velocity_bins = np.linspace(-4.0, 4.0, 18)

    return [
        cart_position_bins,
        cart_velocity_bins,
        pole_angle_bins,
        pole_angular_velocity_bins
    ]


# ------------------------------------------------------------
# Function: discretize_state
# Purpose:
# Convert a continuous CartPole observation into a discrete state.
# ------------------------------------------------------------
def discretize_state(observation, bins):
    """
    Convert the environment observation into a Q-table state.

    Args:
        observation: The four continuous values from CartPole.
        bins: The bins created by create_bins().

    Returns:
        A tuple such as (5, 4, 9, 8).
    """

    discrete_state = []

    for i in range(len(observation)):
        bin_index = np.digitize(observation[i], bins[i])
        discrete_state.append(bin_index)

    return tuple(discrete_state)


# ------------------------------------------------------------
# Function: choose_action
# Purpose:
# Choose an action using epsilon-greedy decision-making.
# ------------------------------------------------------------
def choose_action(q_table, state, epsilon, env):
    """
    Choose an action.

    With probability epsilon, the agent explores randomly.
    Otherwise, the agent chooses the action with the best Q-value.

    Args:
        q_table: Table of learned Q-values.
        state: Current discrete state.
        epsilon: Exploration probability.
        env: CartPole environment.

    Returns:
        0 for left or 1 for right.
    """

    if random.random() < epsilon:
        return env.action_space.sample()

    return np.argmax(q_table[state])


# ------------------------------------------------------------
# Function: moving_average
# Purpose:
# Smooth reward results so the chart is easier to interpret.
# ------------------------------------------------------------
def moving_average(data, window_size):
    """
    Calculate a moving average.

    Args:
        data: List of episode rewards.
        window_size: Number of episodes per average.

    Returns:
        Smoothed reward values.
    """

    data_array = np.array(data)
    return np.convolve(data_array, np.ones(window_size) / window_size, mode="valid")


# ------------------------------------------------------------
# Function: train_agent
# Purpose:
# Train the Q-learning agent.
# ------------------------------------------------------------
def train_agent():
    """
    Train the CartPole Q-learning agent.

    Returns:
        q_table: The trained Q-table.
        rewards_per_episode: Reward from each training episode.
    """

    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    bins = create_bins()

    # Each bin group creates len(bins) + 1 possible values.
    state_space_size = tuple(len(bin_group) + 1 for bin_group in bins)

    # CartPole has two actions: push left or push right.
    action_space_size = env.action_space.n

    # Create the Q-table.
    q_table = np.zeros(state_space_size + (action_space_size,))

    # More episodes gives the Q-table more chances to improve.
    episodes = 30000

    # Learning rate.
    learning_rate = 0.12

    # Discount factor. High value because future rewards matter in CartPole.
    discount_factor = 0.99

    # Start with full exploration.
    epsilon = 1.0

    # Keep a small amount of exploration.
    min_epsilon = 0.01

    # Slow decay helps the agent explore longer.
    epsilon_decay = 0.9997

    rewards_per_episode = []

    # Track the best recent average so we know how well the agent did.
    best_average_reward = 0
    best_q_table = q_table.copy()

    for episode in range(episodes):

        observation, info = env.reset(seed=SEED + episode)
        state = discretize_state(observation, bins)

        done = False
        total_reward = 0

        while not done:

            action = choose_action(q_table, state, epsilon, env)

            next_observation, reward, terminated, truncated, info = env.step(action)

            next_state = discretize_state(next_observation, bins)

            done = terminated or truncated

            # Penalize the agent if the episode ended because the pole fell
            # or the cart moved too far. This gives stronger feedback.
            if terminated:
                reward = -10

            current_q = q_table[state + (action,)]

            if done:
                target_q = reward
            else:
                best_future_q = np.max(q_table[next_state])
                target_q = reward + discount_factor * best_future_q

            new_q = current_q + learning_rate * (target_q - current_q)

            q_table[state + (action,)] = new_q

            state = next_state

            # For reporting, still count the actual survival time.
            total_reward += 1

        rewards_per_episode.append(total_reward)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 1000 == 0:
            recent_average = np.mean(rewards_per_episode[-100:])

            if recent_average > best_average_reward:
                best_average_reward = recent_average
                best_q_table = q_table.copy()

            print(
                f"Episode {episode + 1}: "
                f"Average Reward Last 100 Episodes = {recent_average:.2f}, "
                f"Best Average = {best_average_reward:.2f}, "
                f"Epsilon = {epsilon:.3f}"
            )

            # CartPole-v1 max score is 500.
            # If the recent average is high enough, we can stop early.
            if recent_average >= 475:
                print("\nSolved early. Recent average reward reached 475 or higher.")
                break

    env.close()

    return best_q_table, rewards_per_episode



# ------------------------------------------------------------
# Function: plot_rewards
# Purpose:
# Save a chart of training results.
# ------------------------------------------------------------
def plot_rewards(rewards_per_episode):
    """
    Plot raw rewards and moving average rewards.

    Args:
        rewards_per_episode: List of rewards from training.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(rewards_per_episode, alpha=0.30, label="Episode Reward")

    if len(rewards_per_episode) >= 100:
        average_rewards = moving_average(rewards_per_episode, 100)
        plt.plot(
            range(99, len(rewards_per_episode)),
            average_rewards,
            label="100-Episode Moving Average"
        )

    plt.title("CartPole Q-Learning Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("cartpole_training_rewards.png")
    plt.show()


# ------------------------------------------------------------
# Function: test_agent
# Purpose:
# Test the trained agent.
# ------------------------------------------------------------
def test_agent(q_table):
    """
    Test the trained Q-table.

    Args:
        q_table: The trained Q-table.
    """

    env = gym.make("CartPole-v1", render_mode="human")
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    bins = create_bins()

    test_episodes = 10
    test_rewards = []

    for episode in range(test_episodes):

        observation, info = env.reset(seed=SEED + 1000 + episode)
        state = discretize_state(observation, bins)

        done = False
        total_reward = 0

        while not done:

            action = np.argmax(q_table[state])

            next_observation, reward, terminated, truncated, info = env.step(action)

            next_state = discretize_state(next_observation, bins)

            done = terminated or truncated

            state = next_state

            total_reward += reward

        test_rewards.append(total_reward)

        print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()

    print(f"\nAverage Test Reward: {np.mean(test_rewards):.2f}")
    print(f"Best Test Reward: {np.max(test_rewards):.2f}")


# ------------------------------------------------------------
# Function: main
# Purpose:
# Run training, plot results, and test the agent.
# ------------------------------------------------------------
def main():
    """
    Main program.
    """

    print("Starting CartPole Q-learning training...")

    q_table, rewards_per_episode = train_agent()

    print("\nTraining complete.")

    final_average_reward = np.mean(rewards_per_episode[-100:])

    print(f"Final Average Reward Last 100 Episodes: {final_average_reward:.2f}")

    plot_rewards(rewards_per_episode)

    print("\nTesting trained agent...")

    test_agent(q_table)


if __name__ == "__main__":
    main()