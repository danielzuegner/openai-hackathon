import sys, math
from math import *
import numpy as np

import colorsys

from matplotlib import  pyplot as plt

import sys
sys.path.append('..')

from gym_capturethehack.Agent import Agent
from gym_capturethehack.config import config
from gym_capturethehack.config import Mode
from gym_capturethehack.EnvironmentManager import EnvironmentManager
from gym_capturethehack.AgentState import AgentState
from gym_capturethehack.Observation import Observation

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener,
                      circleShape)

import gym
from gym import spaces
from gym.utils import colorize, seeding
from gym.spaces import MultiDiscrete, Tuple

import pyglet
from pyglet import gl

from enum import Enum
class EntityType(Enum):
    AGENT = 1
    OBSTACLE = 2
    BULLET = 3

STATE_W = config["image_size"][0]
STATE_H = config["image_size"][1]
UPSCALE_FACTOR = 7
PLAYFIELD = (STATE_W + STATE_H) / 5

FPS = 7

class ContactListener(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.agents = []
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):

        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData

        if 'class' not in u1 or 'class' not in u2:
            return None

        if u1['class'] == EntityType.AGENT:
            if u2['class'] == EntityType.BULLET:
                if 'agent' in u1 and 'agent' in u2:
                    self.env.kill(u2['agent'], u1['agent'])
                    u2['toBeDestroyed'] = True

        if u2['class'] == EntityType.AGENT:
            if u1['class'] == EntityType.BULLET:
                if 'agent' in u1 and 'agent' in u2:
                    self.env.kill(u1['agent'], u2['agent'])
                    u1['toBeDestroyed'] = True

        if u1['class'] == EntityType.BULLET:
            if u2['class'] == EntityType.OBSTACLE:
                u1['toBeDestroyed'] = True

        if u2['class'] == EntityType.BULLET:
            if u1['class'] == EntityType.OBSTACLE:
                u2['toBeDestroyed'] = True

        return None

def drawCircle(body, factor, color = (0.4, 0.8, 0.4, 1.0), numPoints = 10):
    gl.glBegin(gl.GL_POLYGON)
    gl.glColor4f(color[0], color[1], color[2], color[3])
    radius = body.fixtures[0].shape.radius
    pos = body.position
    for i in range(numPoints):
        angle = radians(float(i) / numPoints * 360.0)
        x = radius * cos(angle) + pos[0]
        y = radius * sin(angle) + pos[1]
        gl.glVertex3f(x * factor, y * factor, 0)
    gl.glColor4f(0.4, 0.9, 0.4, 1.0)
    gl.glEnd()

def drawOrientationIndicator(body, factor, color, length=2):
    angle = body.angle
    pos = body.position
    body_radius = body.fixtures[0].shape.radius
    len = body_radius + length
    x = cos(angle)
    y = sin(angle)
    gl.glBegin(gl.GL_LINES)
    gl.glColor4f(color[0], color[1], color[2], color[3])
    gl.glVertex3f(factor * pos[0], factor * pos[1], 0)
    gl.glVertex3f(factor * (pos[0] + len * x), factor * (pos[1] + len * y), 0)
    gl.glEnd()


class CaptureTheHackEnv(gym.Env):
    metadata = {
        'render.modes': ['human','state_pixels'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):


        self._seed()
        self.collisionDetector= ContactListener(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.collisionDetector)
        self.viewer = None
        self.agents = []

        if config['mode'] == Mode.TEAM_DEATHMATCH:
            n_agents = sum(config['team_counts'])

            for id, agents in enumerate(config["team_counts"]):
                for agent in range(agents):
                    agent_object = Agent(team=id, id=agent, learning=False)
                    self.agents.append(agent_object)
        elif config['mode'] == Mode.TEST_SETUP:
            n_agents = 2
            learning_agent = Agent(team=0, id=0, learning=False)
            learning_agent.behavior = Agent.Behavior.LEARNING

            do_nothing_agent = Agent(team=1, id=0, learning=False)
            do_nothing_agent.behavior = Agent.Behavior.DO_NOTHING
            self.agents.append(learning_agent)
            self.agents.append(do_nothing_agent)

        self.env_manager = EnvironmentManager()
        self.teams_members_alive = None
        self.n_teams = 0
        self.agent_radius = 0
        self.total_steps = 0


        lower_dims = np.ones([n_agents, 2]) * -1
        upper_dims = np.ones([n_agents, 2]) * 1
        self.action_space = Tuple([spaces.Box( lower_dims, upper_dims ),             # (accelerate, decelerate), (steer left, steer right)
                                  MultiDiscrete([[0,1] for _ in range(n_agents)]),   # shoot or do not
                                  MultiDiscrete([[0,1] for _ in range(n_agents)])])  # communication bit

        min_reward = config["min_reward"]
        max_reward = config["max_reward"]


        b = spaces.Box(low=0, high=255, shape=(n_agents, STATE_H, STATE_W, 3))
        c = spaces.Box(low=min_reward, high=max_reward, shape=(n_agents, ))

        self.observation_space = Tuple((b,c))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):

        for body in self.world.bodies:
            self.world.DestroyBody(body)
        return None

    def _reset(self):
        self.done = False
        self._destroy()
        for agent in self.agents:
            agent.body = None
        self.human_render = False
        self.time = 0.0
        self._create_world()
        self.teams_members_alive = list(config['team_counts'])
        self.n_teams = len(self.teams_members_alive)

        self._render('human')
        obs = self._render('state_pixels')
        output = self.env_manager.observation_to_observation_space(obs)

        return output

    def _step(self, action):

        self.total_steps += 1
        if (self.total_steps) % 5000 == 0:
            for agent in self.agents:
                print("**************************")
                print("Agent {} Team {}: {} kills, {} deaths".format(agent.id, agent.team, agent.kills, agent.deaths))
                print("**************************")

        for agent in self.agents:
            agent.reward = 0

        for body in self.world.bodies:
            if 'toBeDestroyed' in body.userData and body.userData['toBeDestroyed'] == True:
                self.world.DestroyBody(body)
        for ix, agent in enumerate(self.agents):
            if not agent.is_alive or len(agent.body.fixtures) == 0:
                continue
            self.env_manager.split_actions(action)
            _action = self.env_manager.get_agent_action(agent.team, agent.id)
            movement = np.array([_action[0], _action[1]])
            shoot = _action[2]
            communication_bit = _action[3]

            body = agent.body
            angle = body.angle
            pos = body.position
            x = cos(angle)
            y = sin(angle)
            sq = self.agent_radius * self.agent_radius
            # acceleration / deceleration
            # mass is computed as pi * r^2
           # agent.body.ApplyLinearImpulse(((movement[0] + x)/2*sq, (movement[0] + y)/2*sq), body.worldCenter, True)
            agent.body.ApplyLinearImpulse(((movement[0] * x)/2*sq, (movement[0] * y)/2*sq), body.worldCenter, True)
            # steering
            # inertia is computed as pi/2 * r^4
            agent.body.ApplyAngularImpulse((movement[1])/6*sq*sq, True)
            # communication

            agent.body.userData['communicate'] = communication_bit

            agent.reward += config['per_timestep_reward']

            if shoot == 1:
                self.agentShoot(agent)

        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.time += 1.0 / FPS

        obs = self._render('state_pixels')
        output = self.env_manager.observation_to_observation_space(obs)

        if self.time * FPS >= config['episode_time_step_limit']:
            self.done = True

        return output, 0, self.done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'state_pixels':
            WIDTH = STATE_W
            HEIGHT = STATE_H
            factor = UPSCALE_FACTOR
        if mode == 'human':
            WIDTH = STATE_W * UPSCALE_FACTOR
            HEIGHT = STATE_H * UPSCALE_FACTOR
            factor = UPSCALE_FACTOR

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WIDTH, HEIGHT)
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glEnable(gl.GL_MULTISAMPLE)
            gl.glEnable(gl.GL_POLYGON_SMOOTH)
            gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
            self.transform = rendering.Transform()

        win = self.viewer.window
        if mode == 'human':
            win.switch_to()
            win.dispatch_events()

        if not self.human_render:
            win.flip()
        win.clear()
        t = self.transform
        gl.glViewport(0, 0, WIDTH, HEIGHT)
        t.enable()
        self._render_world(WIDTH, HEIGHT, factor)
        t.disable()
        obs = Observation()
        if mode == 'human':
            self.human_render = True
            self.draw_agents(-1,-1, factor)
            win.flip()
        if mode == 'state_pixels':
            for agent in self.agents:
                self.draw_agents(agent.team, agent.id, factor)
                image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
                arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
                arr = arr.reshape(HEIGHT, WIDTH, 4)
                arr = arr[::-1, :, 0:3]
                #print(arr.shape)
                #plt.imshow(arr)
                #plt.show()

                state = AgentState(arr, agent.reward)
                obs.set_agent_state(agent.team, agent.id, state=state)
        return obs


    def _render_world(self, WIDTH, HEIGHT, factor):

        playfield = PLAYFIELD * factor  # Game over boundary
        color = (0.5, 0.5, 0.5)

        upper_y = playfield + STATE_H / 2 * factor
        lower_y = - playfield + STATE_H / 2 * factor
        upper_x = playfield + STATE_W / 2 * factor
        lower_x = -playfield + STATE_W / 2 * factor

        self.viewer.draw_polygon(
            [(lower_x, upper_y), (upper_x, upper_y),
             (upper_x, lower_y), (lower_x, lower_y)],
            color=color)
        self.viewer.draw_line((lower_x, upper_y),(upper_x, lower_y), color=color)

        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        for body in self.world.bodies:
            pos = body.position
            if body.userData['class'] == EntityType.BULLET:
                #gl.glBegin(gl.GL_POLYGON)
                color = config['bullet_color']
                drawCircle(body, factor, color)

            elif type(body.fixtures[0].shape) == Box2D.Box2D.b2PolygonShape:
                gl.glBegin(gl.GL_POLYGON)
                color = (1, 1, 1, 1)
                gl.glColor4f(color[0], color[1], color[2], color[3])
                for point in body.fixtures[0].shape.vertices:
                    gl.glVertex3f((point[0] + pos[0])*factor, (point[1] + pos[1])*factor, 0)
                gl.glEnd()

    def compute_team_colors(self, team_id, agent_id):
        team_colors = list(range(self.n_teams))
        # human mode
        if team_id == -1:
            team_colors = [colorsys.hsv_to_rgb(i * 1.0 / self.n_teams, 0.5, 0.5) + (1,) for i in range(self.n_teams)]
        else:
            for team in range(self.n_teams):
                if team == team_id:
                    team_colors[team] = config['team_color']
                else:
                    team_colors[team] = config['enemy_color']
        return team_colors

    def draw_agents(self, team_id, agent_id, factor):
        team_colors = self.compute_team_colors(team_id, agent_id)
        for agent in self.agents:
            if agent.is_alive == False or len(agent.body.fixtures) == 0:
                continue
            body = agent.body
            if body.userData['class'] == EntityType.AGENT:
                color = None
                if agent.team == team_id and agent.id == agent_id:
                    color = config['self_color']
                else:
                    color = team_colors[agent.team]
                color = list(color)
                if body.userData['communicate'] == 1:
                    for c in range(len(color)):
                        if color[c] >= 1 - config['communication_color_add']:
                            color[c] -=  config['communication_color_add']
                        else:
                            color[c] += config['communication_color_add']

                drawCircle(body, factor, color)
                drawOrientationIndicator(body, factor, color)


    def _create_world(self):

        upper_y = PLAYFIELD + STATE_H / 2
        lower_y = - PLAYFIELD + STATE_H / 2
        upper_x = PLAYFIELD + STATE_W / 2
        lower_x = -PLAYFIELD + STATE_W / 2

        wall_thickness = 2

        upperWallBox =   [(lower_x, + upper_y),
                          (upper_x, + upper_y),
                          (lower_x, + upper_y + wall_thickness),
                          (upper_x, + upper_y + wall_thickness)]
        lowerWallBox =   [(lower_x, + lower_y),
                          (upper_x, + lower_y),
                          (lower_x, + lower_y + wall_thickness),
                          (upper_x, + lower_y + wall_thickness)]
        leftWallBox =  [(lower_x, + upper_y),
                        (lower_x, + lower_y),
                        (lower_x + wall_thickness, + upper_y),
                        (lower_x + wall_thickness, + lower_y)]
        rightWallBox = [(upper_x, + upper_y),
                        (upper_x, + lower_y),
                        (upper_x - wall_thickness, + upper_y),
                        (upper_x - wall_thickness, + lower_y)]

        self.agent_radius = (STATE_H + STATE_W) / 100
        #self.agent_radius = 4

        if config['mode'] == Mode.TEAM_DEATHMATCH:
            self.create_agents_team_deathmatch()
        elif config['mode'] == Mode.TEST_SETUP:
            self.create_agents_test_setup()

        upperWall = self.world.CreateStaticBody()
        upperWall.CreatePolygonFixture(vertices=upperWallBox, density=100000)
        upperWall.userData = {"class": EntityType.OBSTACLE}

        lowerWall = self.world.CreateStaticBody()
        lowerWall.CreatePolygonFixture(vertices=lowerWallBox, density=100000)
        lowerWall.userData = {"class": EntityType.OBSTACLE}

        leftWall = self.world.CreateStaticBody()
        leftWall.CreatePolygonFixture(vertices=leftWallBox, density=100000)
        leftWall.userData = {"class": EntityType.OBSTACLE}

        rightWall = self.world.CreateStaticBody()
        rightWall.CreatePolygonFixture(vertices=rightWallBox, density=100000)
        rightWall.userData = {"class": EntityType.OBSTACLE}


    def create_agents_team_deathmatch(self):
        upper_y = PLAYFIELD + STATE_H / 2
        lower_y = - PLAYFIELD + STATE_H / 2
        upper_x = PLAYFIELD + STATE_W / 2
        lower_x = -PLAYFIELD + STATE_W / 2
        #self.agent_radius = 4

        for agent in self.agents:
            agent_body = self.world.CreateDynamicBody(
                position=(np.random.uniform(low=lower_x + self.agent_radius, high=upper_x - self.agent_radius), np.random.uniform(low=lower_y + self.agent_radius, high=upper_y - self.agent_radius)),
                fixtures=fixtureDef(shape=circleShape(
                    radius=self.agent_radius), density=2),
             )
            agent.body = agent_body
            agent_body.linearDamping = .3
            agent_body.angularDamping = .3
            agent_body.angle = np.random.uniform(low=0, high=2*pi)
            agent_body.userData = {"class": EntityType.AGENT, 'agent': agent, 'last_shot': self.time, 'toBeDestroyed': False, 'communicate': 0}
            agent.is_alive = True
            agent.game_over = False
            agent.reward = 0

    def create_agents_test_setup(self):
        print("test_setup")
        upper_y = PLAYFIELD + STATE_H / 2
        lower_y = - PLAYFIELD + STATE_H / 2
        upper_x = PLAYFIELD + STATE_W / 2
        lower_x = -PLAYFIELD + STATE_W / 2

        learning_agent = self.agents[0]
        l_agent_body = self.world.CreateDynamicBody(
                position=(10, 13),
                fixtures=fixtureDef(shape=circleShape(
                    radius=self.agent_radius), density=2),
             )
        learning_agent.body = l_agent_body
        l_agent_body.linearDamping = .3
        l_agent_body.angularDamping = .3
        l_agent_body.angle = pi
        l_agent_body.userData = {"class": EntityType.AGENT, 'agent': learning_agent, 'last_shot': self.time,
                               'toBeDestroyed': False, 'communicate': 0}

        learning_agent.is_alive = True
        learning_agent.game_over = False
        learning_agent.reward = 0


        do_nothing_agent = self.agents[1]
        d_agent_body = self.world.CreateDynamicBody(
            position=(40, 35),
            fixtures=fixtureDef(shape=circleShape(
                radius=self.agent_radius), density=2),
        )
        do_nothing_agent.body = d_agent_body
        d_agent_body.linearDamping = .3
        d_agent_body.angularDamping = .3
        d_agent_body.angle = pi
        d_agent_body.userData = {"class": EntityType.AGENT, 'agent': do_nothing_agent, 'last_shot': self.time,
                                 'toBeDestroyed': False, 'communicate': 0}

        do_nothing_agent.is_alive = True
        do_nothing_agent.game_over = False
        do_nothing_agent.reward = 0



    def agentShoot(self, agent):
        agentPos = agent.body.position
        agentRotation = agent.body.angle
        x = cos(agentRotation)
        y = sin(agentRotation)
        if agent.is_alive == False or len(agent.body.fixtures) == 0:
            return
        # a new shot is only allowed every 1.0 seconds
        if self.time - agent.body.userData['last_shot'] < 1.0:
            return
        agent.body.userData['last_shot'] = self.time
        bullet_radius = self.agent_radius / 4
        bullet_impulse = 140 * bullet_radius * bullet_radius
        radius = agent.body.fixtures[0].shape.radius
        bullet = self.world.CreateDynamicBody(
            position=(agentPos[0] + (bullet_radius + radius + 1)*x, agentPos[1] + (bullet_radius + radius + 1)*y),
            fixtures=fixtureDef(shape=circleShape(
                radius=bullet_radius), density=1),
        )
        bullet.fixtures[0].sensor = True
        bullet.userData = {'agent': agent, 'class': EntityType.BULLET}
        bullet.ApplyLinearImpulse((bullet_impulse*x,bullet_impulse*y), bullet.worldCenter, True)
        bullet.linearDamping = 0

    def kill(self, agent1, agent2):
        """
        :param agent1: the agent that killed agent2
        :param agent2: the agent killed by agent1
        """
        print("Agent {} in team {} kills Agent {} of team {}".format(agent1.id, agent1.team, agent2.id, agent2.team))
        agent2.body.userData['toBeDestroyed'] = True
        agent2.is_alive = False
        if agent1.team == agent2.team:
            agent1.reward += config['team_kill_punishment']
        else:
            agent1.reward += config['kill_reward']
            self.give_team_reward(agent1.team,config['assist_reward'])
        agent2.reward += config['die_punishment']

        agent1.kills += 1
        agent2.deaths += 1

        self.teams_members_alive[agent2.team] -= 1
        if self.teams_members_alive[agent2.team] == 0:
            self.give_team_reward(agent2.team, config['team_loss_punishment'], True, True)

        team_members_alive_np = np.array(self.teams_members_alive)
        if (team_members_alive_np > 0).sum() == 1:
            self.done = True
            team_alive = np.argmax(team_members_alive_np > 0)
            self.give_team_reward(team_alive, config['team_win_reward'], True, True)

    def give_team_reward(self, team, reward, dead_receive_reward = False, game_over = False):
        for agent in self.agents:
            if agent.is_alive == True or dead_receive_reward:
                if agent.team == team:
                    agent.reward += reward
                    agent.game_over = game_over
