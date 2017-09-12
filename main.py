import gym
from gym_capturethehack.AgentManager import AgentManager
from gym_capturethehack.envs.capturethehack import CaptureTheHackEnv

env = gym.make('CaptureTheHack-v0')
episodes = 200000
steps = 5000

agent_manager = AgentManager()

for i_episode in range(episodes):
    observation = env.reset()

    for t in range(steps):
        env.render("human")
        #print(observation)

        observation_proc = agent_manager.observation_space_to_observation(observation)
        cumulative_action = agent_manager.merge_actions(observation_proc)
        observation, reward, done, info = env.step(cumulative_action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
print(agent_manager.get_team_rewards())
agent_manager.save_sessions()
for team_id, reward in agent_manager.get_team_rewards():
    print("Team " + str(team_id) + " has obtained a total reward of: " + str(reward))
