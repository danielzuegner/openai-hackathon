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

    def merge_actions(self, observation):
        """
            Asks every agent to decide on the next step and combines the given actions for complying with the OpenAI Gym API

            observation: Observation object that includes the states, rewards, and communication information for every agent

            returns a numpy array with shape (number of agents, 2) that contains actions for every agent
        """
        actions = np.zeros((len(self.teams_agents, 2))) # two action possibilities per agent (go forth/back, rotation to left/right)

        for team_id, agent_id in self.teams_agents:
            agent = self.teams_agents[(team_id, agent_id)]
            actions[team_id, :] = agent.get_next_action(observation.get_agent_state(team_id, agent_id))

        return actions