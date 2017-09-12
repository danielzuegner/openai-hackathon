from gym_capturethehack.config import config
from gym_capturethehack.QLearner import QLearner
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(config["logging_level"])
from enum import Enum

class Agent:

    class Behavior(Enum):
        LEARNING = 1
        DO_NOTHING = 2
        RANDOM = 3

    def __init__(self, team, id, learning=True):
        self.team = team
        self.id = id
        self.body = None
        self.is_alive = True
        if learning:
            self.learner = QLearner(config["team_counts"][self.team], self.team,self.id)
        else:
            self.learner = None
        self.reward = 0
        self.game_over = False # game is over for the team
        self.kills = 0
        self.deaths = 0
        self.behavior = Agent.Behavior.LEARNING

    def get_next_action(self, agent_state):
        if self.behavior == Agent.Behavior.LEARNING:
            return self.get_action_Q(agent_state)
        elif self.behavior == Agent.Behavior.DO_NOTHING:
            return (0,0,0,0)
        elif self.behavior == Agent.Behavior.RANDOM:
            return np.random.choice(self.learner.actions)

    def get_action_Q(self, agent_state):
        if not self.is_alive:
            if self.game_over:
                self.reward += agent_state.reward
            return (0, 0, 0, 0)
        if self.game_over:
            self.reward += agent_state.reward
        prev_qs = np.squeeze(np.array(self.learner.previous_q))

        #print("..")
        #print(np.squeeze(np.array(prev_qs)).shape)
        prev_action = self.learner.previous_action

        frame = np.expand_dims(agent_state.frame, axis=0)
        Qout, action = self.learner.inference(frame)

        Qmax = np.max(Qout)
        targetQ = prev_qs
        targetQ[prev_action] = agent_state.reward + self.learner.y * Qmax

        self.reward += agent_state.reward

        self.learner.optimize(frame, targetQ)
        return action
        #return (0.5, -0.5, 1, 1) # action

    def print_weight_statistics(self):
        if self.learner is not None:
            self.learner.print_statistics()

    def save_session(self, path=""):
        if self.learner is not None:
            self.learner.save_session(path)
       