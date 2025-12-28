import numpy as np
from numpy.linalg import norm

np.random.seed(42)

def Large(w):
    TINY = 1e-14
    R = np.zeros((3, 3))
    t = norm(w)
    if abs(t) < TINY:
        return np.identity(3)

    w = w/t
    st = np.sin(t)
    ct = np.cos(t)
    vt = 1.0 - ct
    t2, t1, t0 = w * st
    w0, w1, w2 = w

    R[0][0]=w0 * w0 * vt + ct
    R[1][0]=w0 * w1 * vt + t0 
    R[2][0]=w0 * w2 * vt - t1 
    R[0][1]=w0 * w1 * vt - t0 
    R[1][1]=w1 * w1 * vt + ct
    R[2][1]=w1 * w2 * vt + t2
    R[0][2]=w0 * w2 * vt + t1
    R[1][2]=w1 * w2 * vt - t2
    R[2][2]=w2 * w2 * vt + ct

    return R

def log(R):

    theta = np.arccos((R[0][0]+R[1][1]+R[2][2]-1)/2)
    sin = np.sin(theta)
    v1 = (R[2][1] - R[1][2])/(2*sin)
    v2 = (R[0][2] - R[2][0])/(2*sin)
    v3 = (R[1][0] - R[0][1])/(2*sin)

    rv = np.array([v1,v2,v3])

    return theta, [v1,v2,v3]

def method1():
    w = np.random.rand(3, 1)
    w = w/norm(w)
    t = lambda x: np.random.rand(x) * np.pi
    w = w*t(1)

    return Large(w)

def method2():
    t = np.random.rand()*2*np.pi

    st = np.sin(t)
    ct = np.cos(t)

    v = np.random.rand(3)
    v = v/norm(v)

    x = np.array([1, 0, 0], dtype='float32') - v
    x = x/norm(x)

    R = (2*x*np.reshape(x, (3, 1)) - np.identity(3))
    mat = [[1, 0, 0],
           [0, ct, -st],
           [0, st, ct]]
    return np.matmul(R, mat)


def method3():
    R = np.random.rand(3, 3)
    R[:, 0] = R[:, 0] / norm(R[:, 0])
    R[:, 1] = R[:, 1] - (np.reshape(R[:, 0], (3,1)) * R[:, 1]) @ R[:, 0]
    R[:, 1] = R[:, 1] / norm(R[:, 1])
    R[:, 2] = np.cross(R[:, 0], R[:, 1])
    R[:, 2] = R[:, 2] / norm(R[:, 2])
    return R

def generate_rotmat(t, axis=2):
    """
    Generates rotation matrix that rotates [t] degrees about axis. (0: x-axis, 1: y-axis, 2: z-axis)
    t : int (degree)
    axis : int (0 : x, 1 : y, 2 : z), default = 2
    """
    w = np.array([0., 0., 0.])
    try:
        w[axis] = 1.
    except:
        return np.identity(3)
    w.reshape((3, 1))
    t = np.radians(t)
    w = w*t

    return Large(w)

def generate_rotmat(t, axis=2):
    """
    Generates rotation matrix that rotates [t] degrees about axis. (0: x-axis, 1: y-axis, 2: z-axis)
    t : int (degree)
    axis : int (0 : x, 1 : y, 2 : z), default = 2
    """
    w = np.array([0., 0., 0.])
    try:
        w[axis] = 1.
    except:
        return np.identity(3)
    w.reshape((3, 1))
    t = np.radians(t)
    w = w*t

    return Large(w)
