"""
@author: Adrià Mompó Alepuz

File used for raw simulation (just run the file) with a hard-coded simple policy.
For a more detailed description of the file see the 'environment.py' header.
"""

import numpy as np
from math import sqrt,acos,trunc,cos,sin
import tkinter as tk
import time


import matplotlib.patches as patch
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

from IPython.display import HTML

# SIMULATION ELEMENTS


class Point:

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def distance2(self,p):
        return (self.x - p.x)*(self.x - p.x) + (self.y - p.y)*(self.y - p.y)

    def _distance2(self,coord):
        return (self.x - coord[0])*(self.x - coord[0]) + (self.y - coord[1])*(self.y - coord[1])
    
    def distance(self,p):
        return sqrt(self.distance2(p))

    def _distance(self,coord):
        return sqrt(self._distance2(coord))
        
    def closest(self, a, b):
        da = b.y - a.y
        db = a.x - b.x
        c1 = da*a.x + db*a.y
        c2 = -db*self.x + da*self.y
        det = da*da + db*db
        cx = 0
        cy = 0
    
        if (det != 0):
            cx = (da*c1 - db*c2) / det
            cy = (da*c2 + db*c1) / det
        else:
            # The point is already on the line
            cx = self.x
            cy = self.y
    
        return Point(cx, cy)


class Unit(Point):

    def __init__(self,x,y,tag,r,vx=0,vy=0):
        Point.__init__(self,x,y)
        self.tag = tag
        self.r = r
        self.vx = vx
        self.vy = vy
        
    def collision(self,u):
        # Square of the distance
        dist = self.distance2(u)
    
        # Sum of the radii squared
        sr = (self.r + u.r)*(self.r + u.r)
    
        # We take everything squared to avoid calling sqrt uselessly. It is better for performances
    
        if (dist < sr):
            # Objects are already touching each other. We have an immediate collision.
            return Collision(self, u, 0.0)
        
        # Optimisation. Objects with the same speed will never collide
        if (self.vx == u.vx and self.vy == u.vy):
            return None
        
        # We place ourselves in the reference frame of u. u is therefore stationary and is at (0,0)
        x = self.x - u.x
        y = self.y - u.y
        myp = Point(x, y)
        vx = self.vx - u.vx
        vy = self.vy - u.vy
        up = Point(0, 0)
    
        # We look for the closest point to u (which is in (0,0)) on the line described by our speed vector
        p = up.closest(myp, Point(x + vx, y + vy))
    
        # Square of the distance between u and the closest point to u on the line described by our speed vector
        pdist = up.distance2(p)
    
        # Square of the distance between us and that point
        mypdist = myp.distance2(p)
    
        # If the distance between u and self line is less than the sum of the radii, there might be a collision
        if (pdist < sr):
         # Our speed on the line
            length = sqrt(vx*vx + vy*vy)
    
            # We move along the line to find the point of impact
            backdist = sqrt(sr - pdist)
            p.x = p.x - backdist * (vx / length)
            p.y = p.y - backdist * (vy / length)
    
            # If the point is now further away it means we are not going the right way, therefore the collision won't happen
            if (myp.distance2(p) > mypdist):
                return None
            
            pdist = p.distance(myp)
    
            # The point of impact is further than what we can travel in one turn
            if (pdist > length):
                return None
            
            # Time needed to reach the impact point
            t = pdist / length
    
            return Collision(self, u, t)
        
        return None


class Pod(Unit):
    
    def __init__(self,x,y,tag,r,vx,vy,angle,partner=None):
        Unit.__init__(self,x,y,tag,r,vx,vy)
        self.angle = angle
        self.nextCPid = 1
        self.checked = 0
        self.timeout = 100
        self.partner = partner
        self.shield = False
        self.lap = 0
        self.start = True

    def getAngle(self,p):
        d = self.distance(p)
        dx = (p.x - self.x) / d
        dy = (p.y - self.y) / d
    
        # Simple trigonometry. We multiply by 180.0 / PI to convert radiants to degrees.
        a = acos(dx) * 180.0 / np.pi
    
        # If the point I want is below me, I have to shift the angle for it to be correct
        if (dy < 0):
            a = 360.0 - a

        return a
        
        
    def diffAngle(self,p):
        a = self.getAngle(p)

        # To know whether we should turn clockwise or not we look at the two ways and keep the smallest
        # The ternary operators replace the use of a modulo operator which would be slower
        right = a - self.angle if self.angle <= a else 360.0 - self.angle + a
        left = self.angle - a if self.angle >= a else self.angle + 360.0 - a
    
        if (right < left):
            return right
        else:
            # We return a negative angle if we must rotate to left
            return -left
        
        
    def rotate(self,p):
        a = self.diffAngle(p)

        # Can't turn by more than 18° in one turn
        if not self.start:
            if (a > 18.0):
                a = 18.0
            elif (a < -18.0):
                a = -18.0
        else:        
            self.start = False
    
        self.angle += a
    
        # The % operator is slow. If we can avoid it, it's better.
        if (self.angle >= 360.0):
            self.angle = self.angle - 360.0
        elif (self.angle < 0.0):
            self.angle += 360.0

    
    def boost(self,thrust):
        # Don't forget that a pod which has activated its shield cannot accelerate for 3 turns
        if (self.shield):
            return
    
        # Conversion of the angle to radiants
        ra = self.angle * np.pi / 180.0
    
        # Trigonometry
        self.vx += cos(ra) * thrust
        self.vy += sin(ra) * thrust
        
        
    def move(self,t):
        self.x += self.vx * t
        self.y += self.vy * t
        
        
    def end(self):
        self.x = round(self.x)
        self.y = round(self.y)
        self.vx = trunc(self.vx * 0.85)
        self.vy = trunc(self.vy * 0.85)
    
        # Don't forget that the timeout goes down by 1 each turn. It is reset to 100 when you pass a checkpoint
        self.timeout -= 1
        
        
    def play(self,p,thrust):
        self.rotate(p)
        self.boost(thrust)
        self.move(1.0)
        self.end()

    
    def bounce(self,u):
        if (isinstance(u,CheckPoint)):
            # Collision with a checkpoint
            self.bounceWithCheckpoint(u)
        else:
            # If a pod has its shield active its mass is 10 otherwise it's 1
            m1 = 10 if self.shield else 1
            m2 = 10 if u.shield else 1
            mcoeff = (m1 + m2) / (m1 * m2)

            nx = self.x - u.x
            ny = self.y - u.y
    
            # Square of the distance between the 2 pods. This value could be hardcoded because it is always 800²
            nxnysquare = nx*nx + ny*ny
    
            dvx = self.vx - u.vx
            dvy = self.vy - u.vy
    
            # fx and fy are the components of the impact vector. product is just there for optimisation purposes
            product = nx*dvx + ny*dvy
            fx = (nx * product) / (nxnysquare * mcoeff)
            fy = (ny * product) / (nxnysquare * mcoeff)
    
            # We apply the impact vector once
            self.vx -= fx / m1
            self.vy -= fy / m1
            u.vx += fx / m2
            u.vy += fy / m2
    
            # If the norm of the impact vector is less than 120, we normalize it to 120
            impulse = sqrt(fx*fx + fy*fy)
            if (impulse < 120.0):
                fx = fx * 120.0 / impulse
                fy = fy * 120.0 / impulse
    
            # We apply the impact vector a second time
            self.vx -= fx / m1
            self.vy -= fy / m1
            u.vx += fx / m2
            u.vy += fy / m2
    
            # This is one of the rare places where a Vector class would have made the code more readable.
            # But this place is called so often that I can't pay a performance price to make it more readable.
    
    def bounceWithCheckpoint(self,u):
        self.nextCPid += 1
        if self.nextCPid==nCPs:
            self.nextCPid = 0
            self.lap += 1
        self.timeout = 100
        

class CheckPoint(Unit):
    pass
    
class Collision:
    def __init__(self,a,b,t):
        self.a = a
        self.b = b
        self.t = t
    

def play(pods, checkpoints):
    # This tracks the time during the turn. The goal is to reach 1.0
    t = 0.0

    collided = [[-1.0,-1.0,-1.0],[-1.0,-1.0],[-1.0]]

    while (t < 1.0):
        firstCollision = None
        col = None
        colID = None
        
        # We look for all the collisions that are going to occur during the turn
        for i in range(len(pods)):
            # Collision with another pod?
            ####print('\npod ',i,' : \n')
            j = i + 1
            while j < len(pods):
                ####print('with ',j,'\n')
                col = pods[i].collision(pods[j])

                # If the collision occurs earlier than the one we currently have we keep it
                if (col != None and col.t + t < 1.0 and (firstCollision == None or col.t < firstCollision.t)):
                    ####print ('collision check\n')
                    if col.t != collided[i][j-i-1]:
                        ####print('collision ',col.t,' ',collided,'\n')
                        firstCollision = col
                        colID = [i,j]
                j += 1

            # Collision with another checkpoint?
            # It is unnecessary to check all checkpoints here. We only test the pod's next checkpoint.
            # We could look for the collisions of the pod with all the checkpoints, but if such a collision happens it wouldn't impact the game in any way
            col = pods[i].collision(checkpoints[pods[i].nextCPid])

            # If the collision happens earlier than the current one we keep it
            if (col != None and col.t + t < 1.0 and (firstCollision == None or col.t < firstCollision.t)):
                firstCollision = col

        if (firstCollision == None):
            # No collision, we can move the pods until the end of the turn
            for i in range(len(pods)):
                pods[i].move(1.0 - t)
        
            # End of the turn
            t = 1.0
        else:
            # Move the pods to reach the time `t` of the collision
            #print('first col ',firstCollision.t)
            for i in range(len(pods)):
                pods[i].move(firstCollision.t)
        
            # Play out the collision
            firstCollision.a.bounce(firstCollision.b)

            # Update collision identifiers only if it is caused by 2 pods
            if colID != None:
                collided[colID[0]][colID[1]-colID[0]-1] = 0.0

            t += firstCollision.t

        
    for i in range(len(pods)):
        pods[i].end()
 
 
    
def test(pods, checkpoints):
    for i in range(len(pods)):
        pods[i].rotate(Point(checkpoints[pods[i].nextCPid].x,
                             checkpoints[pods[i].nextCPid].y))
        pods[i].boost(100)
    
    play(pods, checkpoints)


def collides(newCp,cps):
    col = False
    #print(newCp)
    for cp in cps:
        if cp._distance(tuple(newCp)) < 2400:
            col = True
            break
    return col

# SIMULATION SETUP

# Parameters - Will be randomized
nCPs = 6
nPods = 4

cps =[]
pods = []

        # THIS WILL BE RANDOMIZED
cpCoord = (np.random.randint(1000,15000),np.random.randint(1000,8000))
cps.append(CheckPoint(x=cpCoord[0],y=cpCoord[1],tag=0,r=600))

#cpCoord = [[11485,6059],[9128,1829],[5025,5231]]
for i in range(1,nCPs):
    cpCoord = (np.random.randint(1000,15000),np.random.randint(1000,8000))
    #cpCoord
    while collides(cpCoord,cps):
        cpCoord = (np.random.randint(1000,15000),np.random.randint(1000,8000))
        
    cps.append(CheckPoint(x=cpCoord[0],y=cpCoord[1],tag=i,r=600))


        # This will be calculated according 0 and 1 nodes' positions
podCoord = [[12795,5329],[11922,5816],[11048,6302],[10175,6789]]
for i in range(nPods):
    pods.append(Pod(x=podCoord[i][0],y=podCoord[i][1],tag=i,r=400,vx=0,vy=0,angle=0))

'''for j in range (10):
    t0=time.time()
    for i in range(1000):
        test(pods,cps)
    t1=time.time()
    print(t1-t0)'''
    

# GUI

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal',
                     xlim=(-100, 100), ylim=(-50, 50))
# Elements in animation
podsFig =[]
podsVel = []
podsDir = []
podsAim = []
cpsFig =[]
colors = ['b','c','g','y']
# particles holds the locations of the particles
for i in range(nPods):
    podsFig.append(patch.Circle((pods[i].x/100-80,
                               -(pods[i].y/100-45)),radius=4,color=colors[i]))
    ax.add_patch(podsFig[i])
    _, = ax.plot([], [], '-'+colors[i], ms=6)
    podsVel.append(_)
    _, = ax.plot([], [], 's-'+colors[i], ms=6)
    podsDir.append(_)
    _, = ax.plot([], [], 'o-'+colors[i], ms=6)
    podsAim.append(_)

for i in range(nCPs):
    cpsFig.append(patch.Circle((cps[i].x/100-80,
                              -(cps[i].y/100-45)),radius=6,color='r'))
    ax.add_patch(cpsFig[i])


# rect is the box edge
rect = plt.Rectangle([-80,-45],
                     160,
                     90,
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)
counter = ax.text(-90,40,'')
#print(cpsFig)
#print(podsFig)
#app = cpsFig.copy()
#app.extend(podsFig)
#print('\n',app,'\n')
#app.extend([rect])
#print('\n',app,'\n')

def init():
    """initialize animation"""
    global cpsFig,podsFig, rect, nCPs, nPods

    for j in range(nCPs):
        cpsFig[j].center = cps[j].x/100-80, -(cps[j].y/100-45)
        
    for j in range(nPods):
        podsFig[j].center = pods[j].x/100-80, -(pods[j].y/100-45)
        podsVel[j].set_data([], [])
        podsDir[j].set_data([], [])
        podsAim[j].set_data([], [])
    
    rect.set_edgecolor('none')
    counter.set_text('')
    
    extnd = cpsFig.copy()
    extnd.extend(podsFig)
    extnd.extend(podsVel)
    extnd.extend(podsDir)
    extnd.extend(podsAim)
    extnd.extend([rect])
    return extnd

def animate(i):
    print(i)
    """perform animation step"""
    global cpsFig,podsFig, rect, nCPs, nPods
    test(pods,cps)

    ####print('iteration '+str(i)+'\n')
    
    # update pieces of the animation
    rect.set_edgecolor('k')
    counter.set_text(str(i))
    
    for j in range(nCPs):
        cpsFig[j].center = cps[j].x/100-80, -(cps[j].y/100-45)
 
    for j in range(nPods):
        podsFig[j].center = pods[j].x/100-80, -(pods[j].y/100-45)
        
        podsVel[j].set_data([pods[j].x/100-80, pods[j].x/100-80 + pods[j].vx/10],
                            [-(pods[j].y/100-45), -(pods[j].y/100-45) - pods[j].vy/10])
        
        podsDir[j].set_data([pods[j].x/100-80, pods[j].x/100-80 + 10*cos(pods[j].angle*np.pi/180)],
                            [-(pods[j].y/100-45), -(pods[j].y/100-45) - 10*sin(pods[j].angle*np.pi/180)])

        podsAim[j].set_data([pods[j].x/100-80, pods[j].x/100-80 + 10*cos(pods[j].getAngle(cps[pods[j].nextCPid])*np.pi/180)],
                            [-(pods[j].y/100-45), -(pods[j].y/100-45) - 10*sin(pods[j].getAngle(cps[pods[j].nextCPid])*np.pi/180)])

        #print(pods[j].angle)
        #print(np.sqrt(pods[j].vx*pods[j].vx + pods[j].vy*pods[j].vy))
        #print(pods[j].lap)
    #print('\n')

    extnd = cpsFig.copy()
    extnd.extend(podsFig)
    extnd.extend(podsVel)
    extnd.extend(podsDir)
    extnd.extend(podsAim)
    extnd.extend([rect])
    extnd.extend([counter])
    return extnd

ani = animation.FuncAnimation(fig, animate, frames=100,
                              interval=100, blit=True, init_func=init)

#top.mainloop()
# Simulation

#plt.show()

#HTML(ani.to_html5_video())

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

ani.save('simul.mp4',writer = writer)
