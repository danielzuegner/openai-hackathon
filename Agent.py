from config import config
import logging
logger = logging.getLogger(__name__)
logger.setLevel(config["logging_level"])

class Agent:

    def __init__(self, team):
        self.team = team
        return

    def get_next_action(self):
        return
