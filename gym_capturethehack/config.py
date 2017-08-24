import logging

config = {
    "team_counts": (5, 5),
    "team_color" : (0,0,1,1),
    "enemy_color": (1,0,0,1),
    "self_color": (0,1,0,1),
    "communication_color_add": 0.1,
    "bullet_color": (1,1,0.5,1),
    "logging_level": logging.INFO,
    "image_size": (50,50),
    "actions_categories_counts": 4,
    "die_punishment": -100,
    "team_kill_punishment": -50,
    "kill_reward": 40,
    "assist_reward": 10,
    "team_win_reward": 100,
    "team_loss_punishment" : -100
}
