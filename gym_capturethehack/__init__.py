import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CaptureTheHack-v0',
    entry_point='gym_capturethehack.envs:CaptureTheHackEnv'
    #max_episode_steps = 500   
)
