from gym_capturethehack.config import config
from QLearner import QLearner
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
        team_memebers = config["teams_count"][self.team]
        qlearner = QLearner(team_memebers)

        prev_qs = qlearner.previous_q
        prev_action = qlearner.previous_action

        Qout, action = qlearner.inference(agent_state.frame, agent_state.communication_signals)

        Qmax = np.max(Qout)
        targetQ = prev_qs
        targetQ[0, prev_action[0]] = agent_state.reward + qlearner.y * Qmax

        



    

    def kill(agent1, agent2):
        """
        :param agent1: the agent that killed agent2
        :param agent2: the agent killed by agent1
        """
        print("Agent {} in team {} kills Agent {} of team {}".format(agent1.id, agent1.team, agent2.id, agent2.team))

        agent2.body.userData['toBeDestroyed'] = True
        return None
       