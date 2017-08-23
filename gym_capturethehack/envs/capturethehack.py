import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding


import pyglet
from pyglet import gl



STATE_W = 96   # less than Atari 160x192
STATE_H = 96
UPSCALE_FACTOR = 10

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
        print(mode)
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
        win.flip()
        #if mode == 'human':
        #    win.flip()
        self.viewer.onetime_geoms = []
        return arr


    def _render_world(self, WIDTH, HEIGHT, factor):
        PLAYFIELD = 200 / 6 * factor  # Game over boundary
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD + WIDTH / 2, +PLAYFIELD + HEIGHT / 2, 0)
        gl.glVertex3f(+PLAYFIELD + WIDTH / 2, +PLAYFIELD + HEIGHT / 2, 0)
        gl.glVertex3f(+PLAYFIELD + WIDTH / 2, -PLAYFIELD + HEIGHT / 2, 0)
        gl.glVertex3f(-PLAYFIELD + WIDTH / 2, -PLAYFIELD + HEIGHT / 2, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        gl.glEnd()
        gl.glBegin(gl.GL_QUADS)
        for body in self.world.bodies:
            print(body)
            body.ApplyLinearImpulse((10, 5), body.worldCenter, True)
            pos = body.position
            for point in body.fixtures[0].shape.vertices:
                gl.glVertex3f((point[0] + pos[0])*factor, (point[1] + pos[1])*factor, 0)
        gl.glEnd()

    def _create_world(self):
        self.box = [(0,0),(0, 10) , (10, 10), (10,0)]

        box = self.world.CreateDynamicBody()
        box.CreatePolygonFixture(vertices=self.box, density=0.1)
        print(box)
        #box.ApplyLinearImpulse((10,5), box.worldCenter, True)
        box.userData = {"class":"obstacles"}
