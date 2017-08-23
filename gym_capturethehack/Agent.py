from gym_capturethehack.config import config
import logging
logger = logging.getLogger(__name__)
logger.setLevel(config["logging_level"])

class Agent:

    def __init__(self, team, id):
        self.team = team
        self.id = id
        self.body = None
        return

    def get_next_action(self, agent_state):
        if self.learn_type == "Q":
            return get_action_Q(agent_state)

    def get_action_Q(agent_state):
        return None

    def kill(agent1, agent2):
        """
        :param agent1: the agent that killed agent2
        :param agent2: the agent killed by agent1
        """
        print("Agent {} in team {} kills Agent {} of team {}".format(agent1.id, agent1.team, agent2.id, agent2.team))
        return None
       