import gym
import gym_capturethehack

env = gym.make('CaptureTheHack-v0')
#env = gym.make('CarRacing-v0')

env.reset()
while True:
    #print(env.action_space.sample())
    env.step(env.action_space.sample())
    env.render("human")