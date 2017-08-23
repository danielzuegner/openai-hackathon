import gym
import gym_capturethehack

env = gym.make('CaptureTheHack-v0')
#env = gym.make('CarRacing-v0')

env.reset()
while True:
    env.step(None)
    env.render()