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


import pyglet
from pyglet import gl

from enum import Enum
class EntityType(Enum):
    AGENT = 1
    OBSTACLE = 2
    BULLET = 3


STATE_W = config["image_size"][0]
STATE_H = config["image_size"][1]
UPSCALE_FACTOR = 10
PLAYFIELD = (STATE_W + STATE_H) / 5

FPS = 50

class ContactListener(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
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
                    Agent.kill(u2['agent'], u1['agent'])
                    u2['toBeDestroyed'] = True

        if u2['class'] == EntityType.AGENT:
            if u1['class'] == EntityType.BULLET:
                if 'agent' in u1 and 'agent' in u2:
                    Agent.kill(u1['agent'], u2['agent'])
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
        self._seed()
        self.collisionDetector= ContactListener(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.collisionDetector)

        self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))  # steer, gas, brake
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3))
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        return None

    def _reset(self):
        self._destroy()
        self.t = 0.0
        self._create_world()
        return None

    def _step(self, action):
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0 / FPS
        for body in self.world.bodies:
            if 'toBeDestroyed' in body.userData and body.userData['toBeDestroyed'] == True:
                self.world.DestroyBody(body)

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
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(HEIGHT, WIDTH, 4)
            arr = arr[::-1, :, 0:3]
            plt.imshow(arr)
            plt.show()
        return arr


    def _render_world(self, WIDTH, HEIGHT, factor):
        playfield = PLAYFIELD * factor  # Game over boundary
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.3, 0.3, 0.3, 1.0)
        gl.glVertex3f(-playfield + WIDTH / 2, +playfield + HEIGHT / 2, 0)
        gl.glVertex3f(+playfield + WIDTH / 2, +playfield + HEIGHT / 2, 0)
        gl.glVertex3f(+playfield + WIDTH / 2, -playfield + HEIGHT / 2, 0)
        gl.glVertex3f(-playfield + WIDTH / 2, -playfield + HEIGHT / 2, 0)
        gl.glColor4f(1, 1, 1, 1.0)
        gl.glEnd()
        for body in self.world.bodies:
            #body.ApplyLinearImpulse((12, 10), body.worldCenter, True)
            pos = body.position

            if body.userData['class'] == EntityType.AGENT:
                color = None
                if body.userData['agent'].team == 0:
                    color = (0.1, .9, .6, 1)
                elif body.userData['agent'].team == 1:
                    color = (1, 0, 1, 1)
                drawCircle(body, factor, color)
                angle = body.angle
                pos = body.position
                x = cos(angle)
                y = sin(angle)
                #gl.glBegin(gl.GL_LINES)
                #gl.glColor4f(color[0], color[1], color[2], color[3])
                #gl.glVertex3f(factor*pos[0], factor*pos[1], 0)
                #gl.glVertex3f(factor*(pos[0] + 3*x), factor*(pos[1] + 3*y), 0)
                #gl.glEnd()

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

        self.box = [(0,0),(0, 10) , (10, 10), (10,0)]

        upperWallBox =   [(-PLAYFIELD + STATE_W / 2, + PLAYFIELD + STATE_H / 2),
                          (+PLAYFIELD + STATE_W / 2, + PLAYFIELD + STATE_H / 2),
                          (-PLAYFIELD + STATE_W / 2, + PLAYFIELD + STATE_H / 2 + 1),
                          (+PLAYFIELD + STATE_W / 2, + PLAYFIELD + STATE_H / 2 + 1)]
        lowerWallBox =   [(-PLAYFIELD + STATE_W / 2, - PLAYFIELD + STATE_H / 2),
                          (+PLAYFIELD + STATE_W / 2, - PLAYFIELD + STATE_H / 2),
                          (-PLAYFIELD + STATE_W / 2, - PLAYFIELD + STATE_H / 2 + 1),
                          (+PLAYFIELD + STATE_W / 2, - PLAYFIELD + STATE_H / 2 + 1)]
        leftWallBox =  [(-PLAYFIELD + STATE_W / 2, + PLAYFIELD + STATE_H / 2),
                        (-PLAYFIELD + STATE_W / 2, - PLAYFIELD + STATE_H / 2),
                        (-PLAYFIELD + STATE_W / 2 + 1, + PLAYFIELD + STATE_H / 2),
                        (-PLAYFIELD + STATE_W / 2 + 1, - PLAYFIELD + STATE_H / 2)]
        rightWallBox = [(+PLAYFIELD + STATE_W / 2, + PLAYFIELD + STATE_H / 2),
                        (+PLAYFIELD + STATE_W / 2, - PLAYFIELD + STATE_H / 2),
                        (+PLAYFIELD + STATE_W / 2 - 1, + PLAYFIELD + STATE_H / 2),
                        (+PLAYFIELD + STATE_W / 2 - 1, - PLAYFIELD + STATE_H / 2)]

        radius = (STATE_H + STATE_W) / 100

        upper_y = PLAYFIELD + STATE_H / 2 - radius
        lower_y = - PLAYFIELD + STATE_H / 2 + radius
        upper_x = PLAYFIELD + STATE_W / 2 - radius
        lower_x = -PLAYFIELD + STATE_W / 2 + radius

        for id, agents in enumerate(config["team_counts"]):
            for agent in range(agents):
                agent_object = Agent(team=id, id = agent)
                agent_body = self.world.CreateDynamicBody(
                    position=(np.random.uniform(low=lower_x, high=upper_x), np.random.uniform(low=lower_y, high=upper_y)),
                    fixtures=fixtureDef(shape=circleShape(
                        radius=radius), density=1),
                 )
                agent_object.body = agent_body
                agent_body.linearDamping = .002
                agent_body.angle = np.random.uniform(low=0, high=2*pi)
                agent_body.userData = {"class": EntityType.AGENT, 'agent': agent_object}
                self.agentShoot(agent_object)

        #agent1_body.angle = pi/4
        #print(agent1_body.fixtures[0].shape)
        #box.ApplyLinearImpulse((12,10), box.worldCenter, True)
        #self.agentShoot(agent)

        #box2 = self.world.CreateStaticBody()
        #box2.CreatePolygonFixture(vertices = [(x[0] + 40, x[1] + 40) for x in self.box], density=1000)
        #box2.userData = {"class": EntityType.OBSTACLE}

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
        radius = agent.body.fixtures[0].shape.radius
        bullet = self.world.CreateDynamicBody(
            position=(agentPos[0] + 2*radius*x, agentPos[1] + 2*radius*y),
            fixtures=fixtureDef(shape=circleShape(
                radius=0.5), density=1),
        )
        bullet.fixtures[0].sensor = True
        bullet.userData = {'agent': agent, 'class': EntityType.BULLET}
        bullet.ApplyLinearImpulse((10*x,10*y), bullet.worldCenter, True)
        bullet.linearDamping = 0


