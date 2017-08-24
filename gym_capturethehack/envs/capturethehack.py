import sys, math
from math import *
import numpy as np

from matplotlib import  pyplot as plt

import sys
sys.path.append('..')

from gym_capturethehack.Agent import Agent
from gym_capturethehack.config import config

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
UPSCALE_FACTOR = 5
PLAYFIELD = (STATE_W + STATE_H) / 5

FPS = 50

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
                    kill(u2['agent'], u1['agent'])
                    u2['toBeDestroyed'] = True

        if u2['class'] == EntityType.AGENT:
            if u1['class'] == EntityType.BULLET:
                if 'agent' in u1 and 'agent' in u2:
                    kill(u1['agent'], u2['agent'])
                    u1['toBeDestroyed'] = True

        if u1['class'] == EntityType.BULLET:
            if u2['class'] == EntityType.OBSTACLE:
                u1['toBeDestroyed'] = True

        if u2['class'] == EntityType.BULLET:
            if u1['class'] == EntityType.OBSTACLE:
                u2['toBeDestroyed'] = True

        return None

def drawCircle(body, factor, color = (0.4, 0.8, 0.4, 1.0), numPoints = 100):
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


class CaptureTheHackEnv(gym.Env):
    metadata = {
        'render.modes': ['human','state_pixels'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):

        n_agents = sum(config['team_counts'])
        self._seed()
        self.collisionDetector= ContactListener(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.collisionDetector)
        self.viewer = None
        self.agents = []

        lower_dims = np.ones([n_agents, 2]) * -1
        upper_dims = np.ones([n_agents, 2]) * 1
        self.action_space = Tuple([spaces.Box( lower_dims, upper_dims ),             # (accelerate, decelerate), (steer left, steer right)
                                  MultiDiscrete([[0,1] for _ in range(n_agents)]),   # shoot or do not
                                  MultiDiscrete([[0,1] for _ in range(n_agents)])])  # communication bit
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3)) # all pixels

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        return None

    def _reset(self):
        self._destroy()
        self.time = 0.0
        self._create_world()
        return None

    def _step(self, action):

        for body in self.world.bodies:
            if 'toBeDestroyed' in body.userData and body.userData['toBeDestroyed'] == True:
                self.world.DestroyBody(body)
        for ix, agent in enumerate(self.agents):
            if agent.is_alive == False or len(agent.body.fixtures) == 0:
                continue
            movement = action[0][ix]
            shoot = action[1][ix]
            communication_bit = action[2][ix]

            body = agent.body
            angle = body.angle
            pos = body.position
            x = cos(angle)
            y = sin(angle)
            # acceleration / deceleration
            agent.body.ApplyLinearImpulse((movement[0] + x, movement[0] + y), body.worldCenter, True)
            # steering
            agent.body.ApplyAngularImpulse(movement[1], True)
            # communication

            agent.body.userData['communicate'] = communication_bit

            if shoot == 1:
                self.agentShoot(agent)


        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.time += 1.0 / FPS

        teams_members_alive = list(range(len(config['team_counts'])))
        for team in teams_members_alive:
            teams_members_alive[team] = 0

        for agent in self.agents:
            if agent.is_alive:
                teams_members_alive[agent.team] += 1

        teams_alive = 0
        for team in teams_members_alive:
            if teams_members_alive[team] > 0:
                teams_alive += 1

        if teams_alive > 1:
            done = False
        else:
            done = True

        return None, 0, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'state_pixels':
            WIDTH = STATE_W
            HEIGHT = STATE_H
            factor = 1
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
        win.switch_to()
        win.dispatch_events()
        win.clear()
        t = self.transform
        arr = None

        gl.glViewport(0, 0, WIDTH, HEIGHT)
        t.enable()
        self._render_world(WIDTH, HEIGHT, factor)
        t.disable()
        if mode == 'human':
            win.flip()
        if mode == 'state_pixels':
            arr = self.viewer.get_array()
            plt.imshow(arr)
            plt.show()
        return arr


    def _render_world(self, WIDTH, HEIGHT, factor):
        playfield = PLAYFIELD * factor  # Game over boundary
        color = (0.5, 0.5, 0.5)
        self.viewer.draw_polygon([(-playfield + WIDTH / 2, +playfield + HEIGHT / 2),(+playfield + WIDTH / 2, +playfield + HEIGHT / 2),
                                         (+playfield + WIDTH / 2, -playfield + HEIGHT / 2),(-playfield + WIDTH / 2, -playfield + HEIGHT / 2)], color = color)
        self.viewer.draw_line((-playfield + WIDTH / 2, +playfield + HEIGHT / 2),(+playfield + WIDTH / 2, -playfield + HEIGHT / 2), color = color)
        for geom in self.viewer.onetime_geoms:
            geom.render()
        for body in self.world.bodies:
            #body.ApplyLinearImpulse((12, 10), body.worldCenter, True)
            pos = body.position

            if body.userData['class'] == EntityType.AGENT:
                color = None
                if body.userData['agent'].team == 0:
                    if body.userData['communicate'] == 0:
                        color = (0, 1, 0, 1)
                    else:
                        color = (0.5, 1, 0.5, 1)
                elif body.userData['agent'].team == 1:
                    if body.userData['communicate'] == 0:
                        color = (1, 0, 1, 1)
                    else:
                        color = (0.7, 0.2, 0.7, 1)
                drawCircle(body, factor, color)
                angle = body.angle
                pos = body.position
                x = cos(angle)
                y = sin(angle)
                gl.glBegin(gl.GL_LINES)
                gl.glColor4f(color[0], color[1], color[2], color[3])
                gl.glVertex3f(factor*pos[0], factor*pos[1], 0)
                gl.glVertex3f(factor*(pos[0] + 3*x), factor*(pos[1] + 3*y), 0)
                gl.glEnd()

            elif body.userData['class'] == EntityType.BULLET:
                #gl.glBegin(gl.GL_POLYGON)
                color = (1,0,0,1)
                drawCircle(body, factor, color)


            elif type(body.fixtures[0].shape) == Box2D.Box2D.b2PolygonShape:
                gl.glBegin(gl.GL_POLYGON)
                color = (1, 1, 1, 1)
                gl.glColor4f(color[0], color[1], color[2], color[3])
                for point in body.fixtures[0].shape.vertices:
                    gl.glVertex3f((point[0] + pos[0])*factor, (point[1] + pos[1])*factor, 0)
                gl.glEnd()


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

        radius = (STATE_H + STATE_W) / 100


        for id, agents in enumerate(config["team_counts"]):
            for agent in range(agents):
                agent_object = Agent(team=id, id = agent)
                agent_body = self.world.CreateDynamicBody(
                    position=(np.random.uniform(low=lower_x + radius, high=upper_x - radius), np.random.uniform(low=lower_y + radius, high=upper_y - radius)),
                    fixtures=fixtureDef(shape=circleShape(
                        radius=radius), density=1),
                 )
                agent_object.body = agent_body
                agent_body.linearDamping = .002
                agent_body.angle = np.random.uniform(low=0, high=2*pi)
                agent_body.userData = {"class": EntityType.AGENT, 'agent': agent_object, 'last_shot': self.time, 'toBeDestroyed': False, 'communicate': 0}
                self.agents.append(agent_object)
                self.agentShoot(agent_object)

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
        bullet_radius = 0.5
        bullet_impulse = 35
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

def kill(agent1, agent2):
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
    agent2.reward += config['die_punishment']

    return None

