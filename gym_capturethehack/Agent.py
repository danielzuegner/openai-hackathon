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
        self.is_alive = True
        #self.learner = QLearner(config["team_counts"][self.team])
        self.reward = 0
        self.game_over = False # game is over for the team

    def get_next_action(self, agent_state):
        return self.get_action_Q(agent_state)

    def get_action_Q(self, agent_state):
        prev_qs = self.learner.previous_q
        prev_action = self.learner.previous_action

        frame = np.expand_dims(agent_state.frame, axis=0)
        Qout, action = self.learner.inference(frame)

        Qmax = np.max(Qout)
        targetQ = prev_qs
        targetQ[0, prev_action[0]] = agent_state.reward + self.learner.y * Qmax

        self.reward += agent_state.reward

        self.learner.optimize(frame, targetQ)
        return action

       