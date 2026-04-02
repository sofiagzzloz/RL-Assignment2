import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D


class RaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode="human",
    ):
        print("init")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=int
        )
        self.is_view = True
        self.pyrace = PyRace2D(self.is_view)
        self.memory = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs = []
        self.pyrace = PyRace2D(self.is_view, mode=mode)
        obs = self.pyrace.observe()
        return np.array(obs), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done = self.pyrace.is_done()
        obs = self.pyrace.observe()
        return (
            np.array(obs),
            reward,
            done,
            False,
            {
                "dist": self.pyrace.car.distance,
                "check": self.pyrace.car.current_check,
                "crash": not self.pyrace.car.is_alive,
            },
        )

    # def render(self, close=False , msgs=[], **kwargs): # gymnasium.render() does not accept other keyword arguments
    def render(self):  # gymnasium.render() does not accept other keyword arguments
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        # print(self.memory) # heterogeneus types
        # np.save(file, self.memory)
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


class RaceEnvV3(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        print("init v3")
        self.action_space = spaces.Discrete(4)
        # 5 radar readings + speed + distance to next checkpoint, all normalized to [0, 1].
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )
        self.is_view = True
        self.msgs = []
        self.pyrace = PyRace2D(
            self.is_view,
            mode=0,
            action_mode="extended",
            observation_mode="continuous",
            reward_mode="shaped",
        )

    def reset(self, seed=None, options=None):
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs = []
        self.pyrace = PyRace2D(
            self.is_view,
            mode=mode,
            action_mode="extended",
            observation_mode="continuous",
            reward_mode="shaped",
        )
        obs = self.pyrace.observe()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done = self.pyrace.is_done()
        obs = self.pyrace.observe()
        return (
            np.array(obs, dtype=np.float32),
            reward,
            done,
            False,
            {
                "dist": self.pyrace.car.distance,
                "check": self.pyrace.car.current_check,
                "crash": not self.pyrace.car.is_alive,
                "speed": self.pyrace.car.speed,
            },
        )

    def render(self):
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs


class RaceEnvV4(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        print("init v4")
        self.action_space = spaces.Discrete(7)
        # 5 radars + speed + checkpoint distance + turn bias + front clearance.
        self.observation_space = spaces.Box(
            low=np.zeros(9, dtype=np.float32),
            high=np.ones(9, dtype=np.float32),
            dtype=np.float32,
        )
        self.is_view = True
        self.msgs = []
        self.pyrace = PyRace2D(
            self.is_view,
            mode=0,
            action_mode="sharp",
            observation_mode="continuous_sharp",
            reward_mode="shaped_sharp",
        )

    def reset(self, seed=None, options=None):
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs = []
        self.pyrace = PyRace2D(
            self.is_view,
            mode=mode,
            action_mode="sharp",
            observation_mode="continuous_sharp",
            reward_mode="shaped_sharp",
        )
        obs = self.pyrace.observe()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done = self.pyrace.is_done()
        obs = self.pyrace.observe()
        return (
            np.array(obs, dtype=np.float32),
            reward,
            done,
            False,
            {
                "dist": self.pyrace.car.distance,
                "check": self.pyrace.car.current_check,
                "crash": not self.pyrace.car.is_alive,
                "speed": self.pyrace.car.speed,
            },
        )

    def render(self):
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs
