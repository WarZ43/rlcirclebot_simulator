import numpy as np
import math
"""
Stores the logic for the physics of the circlebot, this bot starts at 0 velocity and an inputted initial position,
each simulation timestep takes in the target left and right wheel powers and uses dc motor equations and
skid steer/tank drive physics to simulate the robot applying that power for the neext 1/fps seconds,updating
the bot position and velocity.

max_f = maximum force the robot can accelerate a wheel with
max_v = terminal velocity of robot without resistance
krr = coeff of rolling resistance
offset = how far in the x and y direction the wheels are from the center of bot
k_lat = max lateral friction force relative to max_f
k_elast = coefficient of restitution when colliding with walls
"""
class Circlebot:
    def __init__(self, grid, max_f = 12, max_v = 10, krr = 0.1, mass = 10, radius = 0.4, offset = np.array([0.3, 0.3]),
                 init_pos = None, fps = 30, k_lat = 0.6, k_elast = 0.5, stage = 1):
        self.grid = grid
        self.position = np.array([0.5,0.5,math.pi/4])
        # if stage >= 3:
        #         self.position[2] = np.random.uniform(-20, 20, size=(1,))
        # if stage >= 3:
            # self.position[:2] = np.random.uniform(radius,self.grid.size - radius, size=(2,) )
        self.wv = np.array([0,0], dtype = float)
        #wrt to robot heading
        self.v = np.array([0,0,0], dtype = float)
        self.v_global = np.array([0,0,0], dtype = float)
        self.rad = radius
        self.fps = fps
        self.max_f = max_f
        self.max_v = max_v
        self.mass = mass
        self.krr = krr
        self.x_off = offset[0]
        self.y_off = offset[1]
        self.k_lat = k_lat
        self.k_elast = k_elast
        self.ncolls = 0
        self.nsteps = 0
        # print(self.position)
        
    #simulate the robot applying p = [l,r] power(with range (-1.0,1.0)) to the left and right wheels for 1/fps seconds  
    def simulate(self, p):
        self.nsteps+=1
        #force at same power changes with wheel speed
        f = (p - self.wv/self.max_v)
        np.clip(f, -1, 1)
        f*=self.max_f
        #add rolling resistance
        if self.wv[0]>0:
            f[0] -= self.max_f * self.krr
        elif self.wv[0] <0:
            f[0] += self.max_f * self.krr
        if self.wv[1]>0:
            f[1] -= self.max_f * self.krr
        elif self.wv[1] <0:
            f[1] += self.max_f * self.krr
        #add turning skid forces assuming 0 full body lateral skid
        fy = self.k_lat * self.max_f * np.array([-self.v[1] - self.x_off * (self.wv[1]-self.wv[0]) , -self.v[1] + self.x_off * (self.wv[1]-self.wv[0])])
        #calc net force & accel
        net_f = np.array([np.sum(f), np.sum(fy), np.sum(f*np.array([-self.y_off, self.y_off])) + np.sum(fy*np.array([self.x_off, -self.x_off]))])
        net_a = net_f / np.array([self.mass, self.mass, self.mass*self.rad*self.rad*.5])
        
        #update velocities and position
        self.wv += f / self.mass /self.fps
        self.v += net_a / self.fps
        self.v_global = np.array([self.v[0] *math.cos(self.position[2]) - self.v[1]*math.sin(self.position[2])
                                  , self.v[0]*math.sin(self.position[2]) + self.v[1]*math.cos(self.position[2]),
                                  self.v[2]])
        self.position += self.v_global / self.fps
        collisions = self.grid.collided(self.position[0], self.position[1])
        if not collisions is None:
            for i in collisions:
                if np.linalg.norm(i) == 0:
                    self.v_global = np.zeros((3))
                    proj = 0
                elif np.dot(self.v_global[:2],i) > 0:
                    proj = np.dot(self.v_global[:2], i) * i/ (np.linalg.norm(i)**2)
                else:
                    proj = 0
                self.v_global[:2] -= (1+self.k_elast) * proj
                self.ncolls += 3 + abs(np.linalg.norm((1+self.k_elast) * proj))**2
                self.position[:2] -= (1+self.k_elast) * proj / self.fps
            self.v = self.v_global
            self.v = np.array([self.v[0] *math.cos(-self.position[2]) - self.v[1]*math.sin(-self.position[2])
                                  , self.v[0]*math.sin(-self.position[2]) + self.v[1]*math.cos(-self.position[2]),
                                  self.v[2]])
        self.wv = np.array([self.v[0] - self.v[2]*self.rad, self.v[0] + self.v[2]*self.rad])

        
            
                
        

    