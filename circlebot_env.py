import gymnasium as gym
from gymnasium import spaces
import numpy as np
from grid import Grid
from circlebot import Circlebot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import math

"""
Gymnasium environment for circlebot, stage parameter for what stage of curriculum to run
stages patterns are stored in this environment
"""
class CircleBotEnv(gym.Env):
    def __init__(self, stage, render):
        super(CircleBotEnv, self).__init__()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        #shape is 3 dims of bot velocity + distance from target + angle to target + 11 ray projections = 16  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.renderer = render

        if render:
            self.fig = None
            self.ax = None
            self.robot_patch = None
            self.obstacle_patches = []
            self.target_patch = None
        self.prev_dist = 0
        self.last_colls = 0
        self.stage = stage
        self.grid = Grid()
        self.bot = Circlebot(self.grid, stage = self.stage)
        self.radius = self.grid.radius
        
        #clip training sessions so runs that don't reach target don't take forever
        self.limits = [180, 250,510]
        #stage 1: learn to go towards target with obstacles boundaries
        if self.stage <= 2:
            self.bot.position = np.array([0.5,0.5,math.pi/4])
        if self.stage <= 3:
            for i in range(6):
                self.grid.set_obs(i, 2+i)
                self.grid.set_obs(2+i, i)
            self.target_pos = np.array([7.0,7.0])
        #stage 1: learn to avoid obstacles at low frequency
        #stage 2: learn to avoid obstacles at high frequency
        for i in range(5):
            ran = np.random.uniform(-1,1, size = (1,))
            if ran > 0.6 or (self.stage>1 and ran>0.0):
                self.grid.set_obs(1+i,2+i)
            elif ran < -0.6 or (self.stage>1 and ran<-0.2):
                self.grid.set_obs(2+i, 1+i)
        #stage 3: learn to deal with starting angle changes
        if self.stage >= 3:
            self.bot.position[2] == np.random.uniform(-20,20, size = (1,))
      
        
        self.prev_dist = np.linalg.norm(self.target_pos-self.bot.position[:2])
        self.last_colls = 0

    #stage logic couldn't be reused in init because of some pickling errors with subprocenv
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = Grid()
        self.bot = Circlebot(self.grid, stage = self.stage)
        #stage 1: learn to go towards target with obstacles boundaries
        if self.stage <= 2:
            self.bot.position = np.array([0.5,0.5,math.pi/4])
        if self.stage <= 3:
            for i in range(6):
                self.grid.set_obs(i, 2+i)
                self.grid.set_obs(2+i, i)
            self.target_pos = np.array([7.0,7.0])
        #stage 1: learn to avoid obstacles at low frequency
        #stage 2: learn to avoid obstacles at high frequency
        for i in range(5):
            ran = np.random.uniform(-1,1, size = (1,))
            if ran > 0.6 or (self.stage>1 and ran>0.0):
                self.grid.set_obs(1+i,2+i)
            elif ran < -0.6 or (self.stage>1 and ran<-0.2):
                self.grid.set_obs(2+i, 1+i)
        #stage 3: learn to deal with starting angle changes
        if self.stage >= 3:
            self.bot.position[2] == np.random.uniform(-20,20, size = (1,))
        
        self.prev_dist = np.linalg.norm(self.target_pos-self.bot.position[:2])
        self.last_colls = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        left, right = action

        self._simulate(left, right)

        obs = self._get_obs()
        reward, terminated = self._compute_reward()

        #truncate when timesteps exceed limits or when too many collisions happen
        truncated = self.bot.nsteps > self.limits[self.stage-1] or self.bot.ncolls > 600
        #punish heavily for too many collisions truncation
        if truncated and self.bot.ncolls>600:
            reward -=100
        info = {}
        return obs, reward, terminated, truncated, info

    # use matplot to render one frame
    def render(self, mode="human"):
        if self.renderer:
            if self.fig is None: 
                self.fig, self.ax = plt.subplots()
                self.canvas = FigureCanvas(self.fig)
                self.ax.set_xlim(0, len(self.grid.map))
                self.ax.set_ylim(0, len(self.grid.map[0]))
                self.ax.set_aspect("equal")

                # obstacles
                for i in range(len(self.grid.map)):
                    for j in range(len(self.grid.map[i])):
                        if self.grid.map[i][j]:
                            rect = patches.Rectangle((i, j), 1, 1, color="black")
                            self.ax.add_patch(rect)
                            self.obstacle_patches.append(rect)

                # robot
                self.robot_patch = plt.Circle(self.bot.position[:2], self.radius, color="blue")
                self.ax.add_patch(self.robot_patch)

                # target
                self.target_patch = plt.Circle(self.target_pos, 0.3, color="red")
                self.ax.add_patch(self.target_patch)

            # update robot + target position
            self.robot_patch.center = self.bot.position[:2]
            self.target_patch.center = self.target_pos

            if mode == "human":
                plt.pause(0.001)
            elif mode == "rgb_array":
                self.canvas.draw() 
                buf = np.frombuffer(self.canvas.tostring_argb(), dtype=np.uint8)
                w, h = self.canvas.get_width_height()
                img = buf.reshape(h, w, 4) 

                img = img[:, :, 1:] 
                return img

    def _simulate(self, left, right):
        self.bot.simulate(np.array([left, right]))

    def _get_obs(self):
        dx, dy = self.target_pos - self.bot.position[:2]
        rdx = dx * math.cos(self.bot.position[2]) + dy * math.sin(self.bot.position[2])
        rdy = -dx * math.sin(self.bot.position[2]) + dy * math.cos(self.bot.position[2])
        d = np.linalg.norm(np.array([dx, dy]))
        a = math.atan2(rdy, rdx)
        rays = self.grid.ray_projection(self.bot.position)
        return np.array([self.bot.v[0], self.bot.v[1], self.bot.v[2], d, a] + rays, dtype=np.float32)

    #compute reward, adjust penalties and rewards based on what bad behaviors are being seen
    def _compute_reward(self):
        dist = np.linalg.norm(self.target_pos - self.bot.position[:2])
        old_dist = self.prev_dist if hasattr(self, "prev_dist") else dist
        self.prev_dist = dist

        # Progress reward: scaled distance reduction
        progress = old_dist - dist
        reward = progress * 6.0   
        

        # Small time penalty
        # reward -= 0.05
        # reward -= max(np.linalg.norm(self.bot.v[:2])**2/dist *.05,1.2)
        # dx, dy = self.target_pos - self.bot.position[:2]
        # rdx = dx * math.cos(self.bot.position[2]) + dy * math.sin(self.bot.position[2])
        # rdy = -dx * math.sin(self.bot.position[2]) + dy * math.cos(self.bot.position[2])
        # d = np.linalg.norm(np.array([dx, dy]))
        # a = math.atan2(rdy, rdx)
        # reward -= max(abs(a)/d,2)
        
        # Collision penalty
        reward -= (self.bot.ncolls-self.last_colls)*2
        self.last_colls = self.bot.ncolls   
        # Success
        terminated = dist < 0.4
        if terminated:
            reward += 30.0   #
            reward -= np.linalg.norm(self.bot.v[:2])*2   # encourage stopping

        return reward, terminated
