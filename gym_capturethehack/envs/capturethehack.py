import sys, math
from math import *
import numpy as np

#from matplotlib import  pyplot as plt

import sys
sys.path.append('..')

#from Agent import Agent

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener,
                      circleShape)

import gym
from gym import spaces
from gym.utils import colorize, seeding


import pyglet
from pyglet import gl



STATE_W = 96   # less than Atari 160x192
STATE_H = 96
UPSCALE_FACTOR = 10
PLAYFIELD = 40

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
        print(contact)
        print("contact!")
        return None


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
            #plt.imshow(arr)
            #plt.show()
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
            gl.glBegin(gl.GL_POLYGON)
            body.ApplyLinearImpulse((12, 10), body.worldCenter, True)
            pos = body.position
            if type(body.fixtures[0].shape) == Box2D.Box2D.b2CircleShape:
                radius = body.fixtures[0].shape.radius
                numPoints = 100
                for i in range(numPoints):
                    angle = radians(float(i) / numPoints * 360.0)
                    x = radius * cos(angle) + pos[0]
                    y = radius * sin(angle) + pos[1]
                    gl.glVertex3f(x*factor, y*factor, 0)

            elif type(body.fixtures[0].shape) == Box2D.Box2D.b2PolygonShape:
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

        radius = 5
        box = self.world.CreateDynamicBody(
            position=(5, 5),
            fixtures=fixtureDef(shape=circleShape(
                radius=radius), density=1),
        )
        #box.fixtures[0].sensor = True
        box.linearDamping = .002
        box.userData = {"class":"obstacles"}
        print(box.fixtures[0].shape)
        box.ApplyAngularImpulse(-1, True)


        box2 = self.world.CreateStaticBody()
        box2.CreatePolygonFixture(vertices = [(x[0] + 20, x[1] + 20) for x in self.box], density=1000)

        upperWall = self.world.CreateStaticBody()
        upperWall.CreatePolygonFixture(vertices=upperWallBox, density=100000)

        lowerWall = self.world.CreateStaticBody()
        lowerWall.CreatePolygonFixture(vertices=lowerWallBox, density=100000)

        leftWall = self.world.CreateStaticBody()
        leftWall.CreatePolygonFixture(vertices=leftWallBox, density=100000)

        rightWall = self.world.CreateStaticBody()
        rightWall.CreatePolygonFixture(vertices=rightWallBox, density=100000)