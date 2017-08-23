import gym
from AgentManager import AgentManager
from Agent import Agent

def main():
    
    env = gym.make('CaptureTheHack-v0')
    episodes = 20
    steps = 100

    agent_manager = AgentManager()

    for i_episode in range(episodes):
        observation = env.reset()
        
        for t in range(steps):
            env.render()
            print(observation)
            
            cumulative_action = agent_manager.merge_actions(observation)
            observation, reward, done, info = env.step(cumulative_action)
            
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break