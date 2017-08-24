import numpy as np
from config import config

class Observation:
    def __init__(self):
        self.teams = config["team_counts"]
        self.agent_states = dict()

    def get_agent_state(self, team_id, agent_id):
        self.check_ids(team_id, agent_id)

        return self.agent_states[(team_id, agent_id)]
    
    def set_agent_state(self, team_id, agent_id, state=None):
        self.check_ids(team_id, agent_id)

        if state == None:
            print("State cannot be none!")
            exit()
        
        self.agent_states[(team_id, agent_id)] = state

    def check_ids(self, team_id, agent_id):
        if team_id < 0 or team_id > len(self.teams):
            print("Team id was out of bounds!")
            exit()
        
        agents = self.teams[team_id]

        if agent_id < 0 or agent_id > agents:
            print("Agent id was out of bounds")
            exit()

    def __repr__(self):
        stringRepr = None
        for key, value in self.agent_states.items():
            stringRepr += "Agent with ID " + str(key[0]) + " from team with ID " +str(key[0]) + " has reward: " + str(value.reward)

        return stringRepr            