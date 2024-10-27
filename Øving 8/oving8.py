import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action] * (not done)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

def train_agent(agent, env, n_episodes=10000, max_steps=100):
    rewards = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)

            if done:
                if reward == 1.0:
                    modified_reward = 1.0
                else:
                    modified_reward = -1.0
            else:
                modified_reward = -0.01

            agent.learn(state, action, modified_reward, next_state, done)
            state = next_state
            total_reward += modified_reward
            if done:
                break

        rewards.append(total_reward)
        if episode % 500 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    return rewards

def evaluate_agent(agent, env, n_episodes=100, max_steps=100):
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        for step in range(max_steps):
            env.render()
            action = np.argmax(agent.q_table[state, :])
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

def main():
    n_episodes = 10000
    max_steps = 100
    min_epsilon = 0.01
    max_epsilon = 1.0
    epsilon_decay = 0.995
    alpha = 0.1
    gamma = 0.99

    random_map = generate_random_map(size=8)
    print("Generated Solvable FrozenLake Map:")
    for row in random_map:
        print(' '.join(row))
    print()

    env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=False)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, alpha, gamma, max_epsilon, min_epsilon, epsilon_decay)

    # Training phase
    print("Training phase:")
    rewards = train_agent(agent, env, n_episodes, max_steps)
    np.save("frozenlake_q_table.npy", agent.q_table)

    plt.plot(rewards)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # Evaluation phase
    print("Evaluation phase:")
    eval_env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=False, render_mode="human")
    evaluate_agent(agent, eval_env, n_episodes=10, max_steps=max_steps)

    eval_env.close()

if __name__ == "__main__":
    main()
