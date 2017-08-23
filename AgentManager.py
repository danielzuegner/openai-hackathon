from Agent import Agent
import numpy as np
from config import config
from Observation import Observation

import logging
logger = logging.getLogger(__name__)
logger.setLevel(config["logging_level"])

class AgentManager:

    teams_agents = {}

    def __init__(self):
        logger.debug("Initializing Agent Manager")

        for team_id, n in enumerate(config["team_counts"]):
            for agent_id in range(n):
                self.teams_agents[(team_id, agent_id)] = Agent(team_id)

        return

    def merge_actions(self, observation):
        for team_id, agent_id in self.teams_agents.keys():
            agent = self.teams_agents[(team_id, agent_id)]
            agent.get_next_action(observation.get_agent_state(team_id, agent_id), )

        return