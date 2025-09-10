import math
import numpy as np

"""
This class stores the map that the robot travels on, which is a grid with inputtable size that has
square 1m x 1m obstacles which is stored as a 2d bool array. It also stores the ray projection function which acts
as a low res lidar sensor that projects 11 rays equally spaced from [-pi/2,pi/2] relative to the robot's current
heading and returns how far the ray travels uninteruppted to the nearest .05 meters, capped at 20 meters.
"""
class Grid:
    def __init__(self, size = 8):
        self.size = size
        self.map = [[False for i in range(size)] for i in range(size)]
        self.radius = 0.4
    def set_obs(self, x, y):
        self.map[x][y] = True
    
    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size
    
    """
    acts as a low res lidar sensor that projects 11 rays equally spaced from [-pi/2,pi/2] relative to the robot's current
       heading and returns how far the ray can travel unobstructed to the nearest .05 meters, capped at 20 meters.
    """
    def ray_projection(self, pos, max_dist=20.0, step=0.05):
        x, y, h = pos
        rays = []

        for i in range(11):
            angle = h + (i-5) * (math.pi / 10)
            dx = math.cos(angle)
            dy = math.sin(angle)

            dist = 0.0
            hit = False
            while dist < max_dist:
                # step along ray
                rx = x + dx * dist
                ry = y + dy * dist

                # convert to grid coords
                gx, gy = int(rx), int(ry)

                if self.in_bounds(gx, gy) and self.map[gy][gx]:
                    hit = True
                    break
                dist += step

            if hit:
                rays.append(max(0.0, dist - self.radius))
            else:
                rays.append(max_dist)

        return rays
    def collided(self, x, y):
        fx, fy = math.floor(x), math.floor(y)
        ix = [fx]
        if x - fx < self.radius:
            ix.append(fx-1)
        if x - fx > 1 - self.radius:
            ix.append(fx+1)
        iy = [fy]
        if y - fy < self.radius:
            iy.append(fy-1)
        if y - fy > 1 - self.radius:
            iy.append(fy+1)
        collided = False
        cx, cy = [], []
        for i in ix:
            for j in iy:
                if i > self.size-1 or j > self.size-1 or i < 0 or j < 0 or self.map[i][j]:
                    if i != fx and j!=fy:
                        if i < fx:
                            i+=1
                        if j < fy:
                            j+=1
                        if np.linalg.norm(np.array([x - i, y -j])) > self.radius:
                            continue
                    collided = True
                    cx.append(i)
                    cy.append(j)
        ox, oy = [], []
        ans = []
        if collided:
            for i in range(len(cx)):
                if cx[i] > fx:
                    ox.append(cx[i]-x)
                elif cx[i] < fx:
                    ox.append(fx - x)
                else:
                    ox.append(0)
                if cy[i] > fy:
                    oy.append(cy[i]-y)
                elif cy[i] < fy:
                    oy.append(fy-y)
                else:
                    oy.append(0)
            for i in range(len(ox)):
                collision = np.array([ox[i], oy[i]], dtype= float)
                ans.append(collision)
            # print(ans)
            return ans
        return None
                