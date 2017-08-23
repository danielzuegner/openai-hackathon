import numpy as np
from config import config
import logging
logger = logging.getLogger(__name__)
logger.setLevel(config["logging_level"])

class EnvironmentManager:

    agent_actions = {}

    def __init__(self):
        logger.debug("Initializing EnvironmentManager")

    def split_actions(self, actions):
        """
            Splits the combined actions from the AgentManager and saves them for every agent such that the environment can ask for them

            actions: numpy array with the same shape the AgentManager builds
        """
        team_counts = config["team_counts"]
        for team_id, agent_count in enumerate(team_counts):
            for agent_id in range(agent_count):
                self.agent_actions[(team_id, agent_id)] = actions[team_id, :].reshape((1, 2))

    def get_agent_action(self, team_id, agent_id):
        """
            Returns the action for a given agent (defined by team ID and agent ID) as a numpy array with the form (1, 2).
            First entry is the positional force and the second entry is the rotational force.
        """
        return self.agent_actions[(team_id, agent_id)]