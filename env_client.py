import gym
import zmq
import numpy as np
from gym.spaces import Box

class RemoteRLBenchEnv(gym.Env):
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://localhost:5555")

        self.setSpaces()

    def setSpaces(self):
        print('setting space')
        self.socket.send_pyobj({'cmd': 'set_space'})
        spaces = self.socket.recv_pyobj()
        self.observation_space = Box(spaces["observation_space"]["low"], spaces["observation_space"]["high"], spaces["observation_space"]["shape"].astype(int))
        self.action_space = Box(spaces["action_space"]["low"], spaces["action_space"]["high"], spaces["action_space"]["shape"].astype(int))

    def reset(self):
        self.socket.send_pyobj({'cmd': 'reset'})
        return self.socket.recv_pyobj()

    def step(self, action):
        self.socket.send_pyobj({'cmd': 'step', 'action': action})
        obs, reward, terminated, info = self.socket.recv_pyobj()        

        return obs, reward, terminated, info

    def close(self):
        self.socket.send_pyobj({'cmd': 'close'})
        self.socket.recv_pyobj()
