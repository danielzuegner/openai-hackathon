import logging
from enum import Enum
class Mode(Enum):
    TEAM_DEATHMATCH = 1
    TEST_SETUP = 2

config = {
    "mode": Mode.TEST_SETUP,
    "team_counts": (1, 1),
    "team_color" : (0,0,1,1),
    "enemy_color": (1,0,0,1),
    "self_color": (0,1,0,1),
    "communication_color_add": 0.1,
    "bullet_color": (1,1,0.5,1),
    "logging_level": logging.INFO,
    "image_size": (50,50,3),
    "actions_categories_counts": 4,
    "die_punishment": -100/100,
    "team_kill_punishment": -50/100,
    "kill_reward": 40/100,
    "assist_reward": 10/100,
    "team_win_reward": 100/100,
    "team_loss_punishment" : -100/100,
    "per_timestep_reward": -0.1/100,
    "min_reward": -1000000,
    "max_reward": 1000000,
    'episode_time_step_limit': 10000
}
