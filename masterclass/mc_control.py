import gym
import numpy as np
import random
import itertools
import time


from lake_envs import *


def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """
    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render();
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


class MCControl:
    def __init__(self, epsilon, gamma, env, num_states, num_actions):
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions


    def init_agent(self):
        self.policy = np.random.choice(num_actions, num_states)

        self.Q = {}
        self.visit_count = {}

        for state in range(self.num_states):
            self.Q[state] = {}
            self.visit_count[state] = {}
            for action in range(self.num_actions):
                self.Q[state][action] = 0
                self.visit_count[state][action] = 0


    def get_epsilon_greedy_action(self, greedy_action):
        prob = np.random.random()

        if prob < 1 - self.epsilon:
            return greedy_action

        return np.random.randint(0, self.num_actions)


    def generate_episode(self, policy):
        G = 0
        s = env.reset()
        a = self.get_epsilon_greedy_action(policy[s])

        state_action_reward = [(s, a, 0)]
        while True:
            s, r, terminated, _ = env.step(a)
            if terminated:
                state_action_reward.append((s, None, r))
                break
            else:
                a = self.get_epsilon_greedy_action(policy[s])
                state_action_reward.append((s, a, r))

        t = 1
        for _, _, reward in state_action_reward:
            G += self.gamma ** (t - 1) * reward
            t += 1

        return G, state_action_reward[:-1]

    def argmax(self, Q, policy):
        """
        Finds and returns greedy policy.

        Parameters
        ----------
        Q: nested dictionary {state: {action: q value}}

        Returns
        ----------
        policy: The action to take at a given state, list of length num_state

        """
        for state in range(self.num_states):
            best_action = None
            best_value = float('-inf')

            for action, value in Q[state].items():
                if value > best_value:
                    best_value = value
                    best_action = action
            policy[state] = best_action

        return policy

    def evaluate_policy(self, G, visit_counts, step, action):
        self.Q[step][action] += (G - self.Q[step][action]) / visit_counts[step][action]


    def improve_policy(self, Q, policy):
        self.policy = self.argmax(Q, policy)


    def run_mc_control(self, num_episodes):
        self.init_agent()

        for episode in range(num_episodes):
            G, state_action_reward = self.generate_episode(self.policy)
            seen_state_action = set()

            for state, action, _ in state_action_reward:
                #  if we see step and action pair for a first time in episode
                if (state, action) not in seen_state_action:
                    self.visit_count[state][action] += 1

                    self.evaluate_policy(G, self.visit_count, state, action)

                    seen_state_action.add((state, action))

            self.improve_policy(self.Q, self.policy)

        print (f"Finished training RL agent for {num_episodes} episodes!")
