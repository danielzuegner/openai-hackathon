import gym
import gym_capturethehack
#from gym.wrappers.time_limit import TimeLimit

env = gym.make('CaptureTheHack-v0')
#env = gym.make('CarRacing-v0')

# wrapper that stops execution if time or step limit is reached
#wrapper = TimeLimit(env, 600, 30000)

env.reset()
while True: #not env._past_limit():
    #print(env.action_space.sample())
    env.render("human")
    env.step(env.action_space.sample())
