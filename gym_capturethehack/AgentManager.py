from gym_capturethehack.Agent import Agent
import numpy as np
from gym_capturethehack.config import config
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
                self.teams_agents[(team_id, agent_id)] = Agent(team_id, agent_id)

    def merge_actions(self, observation):
        """
            Asks every agent to decide on the next step and combines the given actions for complying with the OpenAI Gym API

            observation: Observation object that includes the states, rewards, and communication information for every agent

            returns a numpy array with shape (number of agents, number of categories of actions) that contains actions for every agent
        """
        move_rot = np.zeros[len(self.teams_agents), 2]
        shoot = []
        comm = []
        
        for i, team_id, agent_id in enumerate(self.teams_agents):
            agent = self.teams_agents[(team_id, agent_id)]
            action = agent.get_next_action(observation.get_agent_state(team_id, agent_id))
            move_rot[i, 0] = action[0]
            move_rot[i, 1] = action[1]
            shoot.append(action[2])
            comm.append(action[3])
        
        actions = (move_rot, shoot, comm)

        return actions