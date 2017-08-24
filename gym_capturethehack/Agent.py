from gym_capturethehack.config import config
from gym_capturethehack.QLearner import QLearner
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(config["logging_level"])

class Agent:

    def __init__(self, team, id):
        self.team = team
        self.id = id
        self.body = None
        self.learner = QLearner(config["team_count"][self.team])
        
    def get_next_action(self, agent_state):
        return self.get_action_Q(agent_state)

    def get_action_Q(self, agent_state):
        prev_qs = self.learner.previous_q
        prev_action = self.learner.previous_action

        frame = np.expand_dims(agent_state.frame, axis=0)
        Qout, action = self.learner.inference(frame, agent_state.communication_signals)

        Qmax = np.max(Qout)
        targetQ = prev_qs
        targetQ[0, prev_action[0]] = agent_state.reward + self.learner.y * Qmax

        self.learner.optimize(frame, agent_state.communication_signals, targetQ)

    def kill(self, agent1, agent2):
        """
        :param agent1: the agent that killed agent2
        :param agent2: the agent killed by agent1
        """
        print("Agent {} in team {} kills Agent {} of team {}".format(agent1.id, agent1.team, agent2.id, agent2.team))

        agent2.body.userData['toBeDestroyed'] = True
        return None
       