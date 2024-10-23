import numpy as np
from numpy.linalg import *
from PIL import Image
from scipy import integrate
from scipy.optimize import fsolve
import math
from utils import stick

def binary_equation(y1,z1,y2,z2):
    def flinear(x):# k:x[0], b:x[1]
        return np.array([x[0]*y1+x[1]-z1,x[0]*y2+x[1]-z2])
    yzlinear = fsolve(flinear,[0,0])
    return yzlinear

def solve_b(a,c,w,locate):
    '''
    find the upper limit of the integral 
    so that the arc length is equal to the width of the sticker
    locate: the highest point is on th right(1) b = -a*c^2, 
                                     left (2) b = -a*(upper-c)^2
    return: b , wn=upper(Width of converted picture)
    '''
    def func(x):
        def f(x):
            return (1 + ((x-c)*(2*a))**2)**0.5
        f = integrate.quad(f,0,x)[0]-w
        return f
    root = 0
    upper = fsolve(func,[root])[0] # The X coordinate when the arc length = w
    if(locate == 1):
        b = -a * (c**2)
    elif(locate == 2):
        b = -a * ((upper-c)**2)
    wn = int(np.floor(upper))
    return b, wn

def solve_a(hsegment,stride):
    '''
    solve 'a' according to Height drop in one step
    '''
    a = -hsegment/(stride)**2
    a = max(a, -1/stride)
    return a

def bilinear_interpolation(img,x,y):
    w,h = img.size
    xset = math.modf(x)
    yset = math.modf(y)
    u, v = xset[0], yset[0]
    x1, y1 = xset[1], yset[1]
    x2 = x1+1 if u>0 and x1+1<w else x1
    y2 = y1+1 if v>0 and y1+1<h else y1
    #x2, y2 = x1+1, y1+1
    p1_1 = np.array(img.getpixel((x1,y1)))
    p1_2 = np.array(img.getpixel((x1,y2)))
    p2_1 = np.array(img.getpixel((x2,y1)))
    p2_2 = np.array(img.getpixel((x2,y2)))

    pix = (1-u)*(1-v)*p1_1 + (1-u)*v*p1_2 + u*(1-v)*p2_1 + u*v*p2_2
    p = tuple(np.round(pix).astype(np.int32))
    return p

def horizontal(sticker,params):
    '''
    transform the picture according to parabola in horizontal direction
    input:
        sticker: Image type
        height: matrix (store height information for each coordinate)
    output:
        hor_sticker
    '''
    w, h = sticker.size
    c, hsegment, stride, locate = params[0],params[1],params[2],params[3]
        
    a = solve_a(hsegment,stride)
    b, wn = solve_b(a,c,w,locate)
    
    top3 = np.ones((h,wn,3))*255
    top4 = np.zeros((h,wn,1))
    newimg = np.concatenate((top3,top4),axis=2)
    newimg = Image.fromarray(np.uint8(newimg))
    
    def f(x):
        return (1 + ((x-c)*(2*a))**2)**0.5
    x_arc = [integrate.quad(f,0,xnow+1)[0]  for xnow in range(wn)]
    z = np.zeros((1,wn))

    def zfunction(x):
        return a * ((x-c)**2) + b

    for i in range(wn):
        x_map =  min(x_arc[i],w-1)
        z[0][i] = zfunction(i)
        for j in range(h):
            y_map = j
            pix = bilinear_interpolation(sticker,x_map,y_map)
            newimg.putpixel((i,j),pix)
            
    return newimg,z

def pitch(newimg,z,theta):
    w,h = newimg.size
    m = np.array([[1,0,0],
                [0,math.cos(theta),-math.sin(theta)],
                [0,math.sin(theta),math.cos(theta)]])
    invm = inv(m)

    x = np.array(range(w))
    y1, y2 = np.ones([1,w])*0, np.ones([1,w])*(h-1)
    first = np.vstack([x,y1,z]).T
    last = np.vstack([x,y2,z]).T
    pfirst = first.dot(m)
    plast = last.dot(m)
    
    hn = int(np.floor(np.max(plast[:,1])) - np.ceil(np.min(pfirst[:,1])))+1
    shifting = np.ceil(np.min(pfirst[:,1]))
    top3n = np.ones((hn,w,3))*255
    top4n = np.zeros((hn,w,1))
    endimg = np.concatenate((top3n,top4n),axis=2)
    endimg = Image.fromarray(np.uint8(endimg))

    start = np.ceil(pfirst[:,1] - shifting)
    stop = np.floor(plast[:,1] - shifting)

    for i in range(w):
        jstart = int(start[i])
        jstop = int(stop[i])
        def zconvert(y):
            parm = binary_equation(pfirst[i][1],pfirst[i][2],plast[i][1],plast[i][2])
            return parm[0]*y + parm[1]

        for j in range(jstart,jstop+1):
            raw_y = j+shifting
            raw_z = zconvert(raw_y)
            mapping = np.array([i,raw_y,raw_z]).dot(invm)
            pix = bilinear_interpolation(newimg,mapping[0],mapping[1])
            endimg.putpixel((i,j),pix)
    return endimg,shifting

def deformation3d(sticker,operate_sticker,magnification,z_buffer,x,y):
    w, h = sticker.size
    
    area = z_buffer[y:y+h,x:x+w]
    index = np.argmax(area)
    highesty = index // area.shape[1]   # Location coordinates of the highest point in the selected area
    highestx = index % area.shape[1]
    locate = 1 if highestx > area.shape[1]/2 else 2 # =1 if the highest point is to the right
    sign = 1 if highesty < area.shape[0]/2 else -1  # =1 if the highest point is on the top(Forward rotation)
    c = highestx
    if (locate==1):
        hsegment = area[highesty][highestx] - area[highesty][0]
        stride = c
    elif(locate==2):
        hsegment = area[highesty][highestx] - area[highesty][area.shape[1]-1]
        stride = w - highestx
    
    #step = 10
    if (sign==1):
        step = max(min(20,area.shape[0]-highesty-1),1)
        partz = area[highesty][highestx] - area[highesty+step][highestx]
        party = step
        theta = min(math.atan(partz/party),math.radians(40))
    elif(sign==-1):
        step = max(min(20,highesty),1)
        partz = area[highesty][highestx] - area[highesty-step][highestx]
        party = step
        theta = max(-1 * math.atan(partz/party),math.radians(-40))
    operate_params = [c*magnification,hsegment,stride*magnification,locate]
    newimg,z = horizontal(operate_sticker,operate_params)
    endimg,shifting = pitch(newimg,z,theta/2)
    sticker=stick.change_sticker(endimg,magnification)

    return sticker,y