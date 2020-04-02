from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
env.render()

agent = Agent(eps=1,gamma=0.9, alpha = 0.7)
avg_rewards, best_avg_reward = interact(env, agent)