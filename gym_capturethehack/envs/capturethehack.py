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
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1200
WINDOW_H = 1000

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
        'render.modes': ['human'],
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
        return None

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()

        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()
        if mode == 'human':
            self.human_render = True
            win.clear()
            t = self.transform
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self._render_world()
            PLAYFIELD = 2000 / 6  # Game over boundary
            gl.glBegin(gl.GL_QUADS)
            gl.glColor4f(0.4, 0.8, 0.4, 1.0)
            gl.glVertex3f(-PLAYFIELD + WINDOW_W / 2, +PLAYFIELD + WINDOW_H / 2, 0)
            gl.glVertex3f(+PLAYFIELD + WINDOW_W / 2, +PLAYFIELD + WINDOW_H / 2, 0)
            gl.glVertex3f(+PLAYFIELD + WINDOW_W / 2, -PLAYFIELD + WINDOW_H / 2, 0)
            gl.glVertex3f(-PLAYFIELD + WINDOW_W / 2, -PLAYFIELD + WINDOW_H / 2, 0)
            #gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
            #gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
            #gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
            #gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
            gl.glColor4f(0.4, 0.9, 0.4, 1.0)
            gl.glEnd()

            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            #self._render_indicators(WINDOW_W, WINDOW_H)
            win.flip()
        self.viewer.onetime_geoms = []


    def _render_world(self):
        print(self.box)


    def _render_indicators(selfself, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W/40.0
        h = H/40.0
        gl.glColor4f(0,0,0,1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5*h, 0)
        gl.glVertex3f(0, 5*h, 0)
        gl.glVertex3f(0, 0, 0)
        gl.glEnd()

    def _create_world(self):
        self.box = [(0,0),(0, 100) , (100, 100), (100,0)]

        self.world.CreateStaticBody(fixtures=fixtureDef(
            shape=polygonShape(vertices=self.box)
        ))