import numpy as np
from parameters import STD

def add_randomness(fnc):
    def inner(self, *xs, dim):
        # return fnc(self, *xs, dim=dim) + np.random.normal(loc=0, scale=std[0])
        return fnc(self, *xs, dim=dim)
    return inner

class LinearSystemSlowCenter:
    def x0(self, xs):
        return 0.8*xs[0]
    def x1(self, xs):
        return 0.8*xs[1]
    def x2(self, xs):
        return 0.8*xs[2]
    def x3(self, xs):
        return 0.8*xs[3]
    def x4(self, xs):
        return 0.8*xs[4]
    def x5(self, xs):
        return 0.8*xs[5]

class LinearSystemFastCenter:
    def x0(self, xs):
        return 0.4*xs[0]
    def x1(self, xs):
        return 0.4*xs[1]
    def x2(self, xs):
        return 0.4*xs[2]
    def x3(self, xs):
        return 0.4*xs[3]
    def x4(self, xs):
        return 0.4*xs[4]
    def x5(self, xs):
        return 0.4*xs[5]

class UnstableNL:
    # def x0(self, xs):
    #     return (-xs[0]/2)**3 + 1.1*xs[1]**2
    # def x1(self, xs):
    #     return 1.5*xs[1]

    def x0(self, xs):
        return 0.2*xs[0]**3
    def x1(self, xs):
        return 0.1*xs[1]**2


## System from: Strategy Synthesis for Partially-known Switched Stochastic Systems
# Linear - 3 Modes
class JacksonMode1:
    def x0(self, xs):
        return 0.4*xs[0] + 0.1*xs[1]
    def x1(self, xs):
        return 0.5*xs[1]

class JacksonMode2:
    def x0(self, xs):
        return 0.4*xs[0] + 0.5*xs[1]
    def x1(self, xs):
        return 0.5*xs[1]

class JacksonMode3:
    def x0(self, xs):
        return 0.4*xs[0]
    def x1(self, xs):
        return 0.5*xs[0] + 0.5*xs[1]

# Non-Linear - 4 Modes
class JacksonMode1NL:
    def x0(self, xs):
        return xs[0] + 0.5 + 0.2*np.sin(xs[1])
    def x1(self, xs):
        return xs[1] + 0.4*np.cos(xs[0])

class JacksonMode2NL:
    def x0(self, xs):
        return xs[0] - 0.5 + 0.2*np.sin(xs[1])
    def x1(self, xs):
        return xs[1] + 0.4*np.cos(xs[0])

class JacksonMode3NL:
    def x0(self, xs):
        return xs[0] + 0.4*np.cos(xs[1])
    def x1(self, xs):
        return xs[1] + 0.5 + 0.2*np.sin(xs[0])

class JacksonMode4NL:
    def x0(self, xs):
        return xs[0] + 0.4*np.cos(xs[1])
    def x1(self, xs):
        return xs[1] - 0.5 + 0.2*np.sin(xs[0])


# Non-Linear - Alternative - 4 Modes

class JacksonMode1NLAlt:
    def x0(self, xs):
        return xs[0] + 0.35 + 0.1*np.sin(xs[1])
    def x1(self, xs):
        return xs[1] + 0.15*np.cos(xs[0]) + 0.05*xs[2]
    def x2(self, xs):
        return 0.3*xs[2]+0.4*xs[3]
    def x3(self, xs):
        return 0.4*xs[3] +0.05*xs[4]
    def x4(self, xs):
        return 0.5*xs[4]

class JacksonMode2NLAlt:
    def x0(self, xs):
        return xs[0] - 0.35 + 0.1*np.sin(xs[1])
    def x1(self, xs):
        return xs[1] + 0.15*np.cos(xs[0]) + 0.05*xs[2]
    def x2(self, xs):
        return 0.3*xs[2]+0.4*xs[3]
    def x3(self, xs):
        return 0.4*xs[3] +0.05*xs[4]
    def x4(self, xs):
        return 0.5*xs[4]

class JacksonMode3NLAlt:
    def x0(self, xs):
        return xs[0] + 0.15*np.cos(xs[1])
    def x1(self, xs):
        return xs[1] + 0.35 + 0.1*np.sin(xs[0]) + 0.05*xs[2]
    def x2(self, xs):
        return 0.3*xs[2]+0.4*xs[3]
    def x3(self, xs):
        return 0.4*xs[3] +0.05*xs[4]
    def x4(self, xs):
        return 0.5*xs[4]

class JacksonMode4NLAlt:
    def x0(self, xs):
        return xs[0] + 0.15*np.cos(xs[1])
    def x1(self, xs):
        return xs[1] - 0.35 + 0.1*np.sin(xs[0]) + 0.05*xs[2]
    def x2(self, xs):
        return 0.3*xs[2]+0.4*xs[3]
    def x3(self, xs):
        return 0.4*xs[3] +0.05*xs[4]
    def x4(self, xs):
        return 0.5*xs[4]

## Artifical high dimensional non-linear systems
class NonLin2Dlin2D_M1:
    def x0(self, xs):
        return 0.6*xs[0] -(0.5 + 0.2*np.sin(xs[1]))
    def x1(self, xs):
        return 0.85*xs[1] +0.4*np.cos(xs[0])

class NonLin2Dlin2D_M2:
    def x0(self, xs):
        return 0.4*(xs[0]-1) + np.sin((xs[1]+0.5))
    def x1(self, xs):
        return 0.5*(xs[1]-1) + np.cos((xs[0]+0.5))

class NonLin2Dlin2D_M3:
    def x0(self, xs):
        return 0.6*xs[0] + 0.5 + 0.2*np.sin(xs[1])
    def x1(self, xs):
        return 0.9*xs[1] + 0.4*np.cos(xs[0])

class NonLin2Dlin2D_M4:
    def x0(self, xs):
        return 0.6*xs[0] -(0.5 + 0.2*np.sin(xs[1]))
    def x1(self, xs):
        return 0.85*xs[1] -0.4*np.cos(xs[0])

## CAR -----------------------------------------------------------------------------------------------------------------
## Parameters
Iz = 1536.7                 # yaw inertia of vehicle body
kf = -128916                # front axle equivalent sideslip stiffness
kr = -85944                 # rear axle equivalent sideslip stiffness
lf = 1.06                   # distance between C.G. and front axle
lr = 1.85                   # distance between C.G. and rear axle
m = 20                      # mass of the vehicle
Ts = 0.1                    # discretization step length was 0.1s / 0.08

## States
# [0, 1, 2,   3, 4, 5]
# [x, y, phi, u, v, omega]

## Inputs
# accelaration:     a = 0
# steering angle:   delta

class Car2DMode:
    def __init__(self, phi):
        self.u = 10
        self.phi = phi
    def x0(self,xs):
        return xs[0] + Ts*self.u*np.cos(self.phi)
    def x1(self,xs):
        return xs[1] + Ts*self.u*np.sin(self.phi)

class Car3DMode:
    def __init__(self, phi):
        self.u = 10
        self.omega = 5
        self.phi = phi
    def x0(self,xs):
        return xs[0] + Ts*self.u*np.cos(xs[2])
    def x1(self,xs):
        return xs[1] + Ts*self.u*np.sin(xs[2])
    def x2(self,xs):
        return xs[2] + (self.phi - xs[2])*Ts*self.omega
        # return 0.*xs[2]+ self.phi

class CarSimpleMode1:
    def __init__(self):
        self.a = 0
        self.omega = 0
    def x0(self,xs):
        return xs[0] + Ts*xs[3]*np.cos(xs[2])
    def x1(self,xs):
        return xs[1] + Ts*xs[3]*np.sin(xs[2])
    def x2(self,xs):
        return xs[2] + Ts*self.omega
    def x3(self,xs):
        return xs[3] + Ts*self.a

class CarMode1:
    def __init__(self):
        self.a = 0
        self.delta = 0
    def x0(self,xs):
        return xs[0] + Ts*(xs[3]*np.cos(xs[2]) - xs[4]*np.sin(xs[2]))
    def x1(self,xs):
        return xs[1] + Ts*(xs[4]*np.cos(xs[2]) + xs[3]*np.sin(xs[2]))
    def x2(self,xs):
        return xs[2] + Ts*xs[5]
    def x3(self,xs):
        return xs[3] + Ts*self.a
    def x4(self,xs):
        return (m*xs[3]*xs[4] + Ts*(lf*kf-lr*kr)*xs[5] - Ts*kf*self.delta*xs[3] - Ts*m*(xs[3]**2)*xs[5]) / (m*xs[3]-Ts*(kf+kr))
    def x5(self,xs):
        return (Iz*xs[3]*xs[5] + Ts*(lf*kf-lr*kr)*xs[4] - Ts*lf*kf*self.delta*xs[3]) / (Iz*xs[3]-Ts*((lf**2)*kf + (lr**2)*kr))


class ImportSystem:
    def __init__(self, system_type: str):
        if system_type == 'linear-slow-center':
            self.system = LinearSystemSlowCenter()
        elif system_type == 'linear-fast-center':
            self.system = LinearSystemFastCenter()

        elif system_type == 'jackson-mode1':
            self.system = JacksonMode1()
        elif system_type == 'jackson-mode2':
            self.system = JacksonMode2()
        elif system_type == 'jackson-mode3':
            self.system = JacksonMode3()

        elif system_type == 'jackson-nl-mode1':
            self.system = JacksonMode1NL()
        elif system_type == 'jackson-nl-mode2':
            self.system = JacksonMode2NL()
        elif system_type == 'jackson-nl-mode3':
            self.system = JacksonMode3NL()
        elif system_type == 'jackson-nl-mode4':
            self.system = JacksonMode4NL()

        elif system_type == 'jackson-nl-mode1-Alt':
            self.system = JacksonMode1NLAlt()
        elif system_type == 'jackson-nl-mode2-Alt':
            self.system = JacksonMode2NLAlt()
        elif system_type == 'jackson-nl-mode3-Alt':
            self.system = JacksonMode3NLAlt()
        elif system_type == 'jackson-nl-mode4-Alt':
            self.system = JacksonMode4NLAlt()


        elif system_type == 'NonLin2Dlin2D-mode1':
            self.system = NonLin2Dlin2D_M1()
        elif system_type == 'NonLin2Dlin2D-mode2':
            self.system = NonLin2Dlin2D_M2()
        elif system_type == 'NonLin2Dlin2D-mode3':
            self.system = NonLin2Dlin2D_M3()
        elif system_type == 'NonLin2Dlin2D-mode4':
            self.system = NonLin2Dlin2D_M4()

        elif system_type == 'car-mode1':
            self.system = CarMode1()

        elif system_type == 'car-simple-mode1':
            self.system = CarSimpleMode1()

        elif system_type == 'car-3d-mode1':
            self.system = Car3DMode(phi=-0.3)
        elif system_type == 'car-3d-mode2':
            self.system = Car3DMode(phi=-0.15)
        elif system_type == 'car-3d-mode3':
            self.system = Car3DMode(phi=0.)
        elif system_type == 'car-3d-mode4':
            self.system = Car3DMode(phi=0.3)
        elif system_type == 'car-3d-mode5':
            self.system = Car3DMode(phi=0.15)
        elif system_type == 'car-3d-mode6':
            self.system = Car3DMode(phi=0.45)
        elif system_type == 'car-3d-mode7':
            self.system = Car3DMode(phi=-0.45)


        elif system_type == 'car-3d-mode8':
            self.system = Car3DMode(phi=-0.1)
        elif system_type == 'car-3d-mode9':
            self.system = Car3DMode(phi=0.)
        elif system_type == 'car-3d-mode10':
            self.system = Car3DMode(phi=0.25)
        elif system_type == 'car-3d-mode11':
            self.system = Car3DMode(phi=0.5)
        elif system_type == 'car-3d-mode12':
            self.system = Car3DMode(phi=0.75)
        elif system_type == 'car-3d-mode13':
            self.system = Car3DMode(phi=-0.2)

        elif system_type == 'car-2d-mode1':
            self.system = Car2DMode(phi=-0.3)
        elif system_type == 'car-2d-mode2':
            self.system = Car2DMode(phi=-0.15)
        elif system_type == 'car-2d-mode3':
            self.system = Car2DMode(phi=0.3)
        elif system_type == 'car-2d-mode4':
            self.system = Car2DMode(phi=0.15)
        elif system_type == 'car-2d-mode5':
            self.system = Car2DMode(phi=0.3)

        elif system_type == 'UnstableNL':
            self.system = UnstableNL()

        else:
            raise AttributeError('No function class defined for system type: {}'.format(system_type))

    @add_randomness
    def generator(self, *xs, dim):
        if dim == 0:
            return self.system.x0(xs)
        elif dim == 1:
            return self.system.x1(xs)
        elif dim == 2:
            return self.system.x2(xs)
        elif dim == 3:
            return self.system.x3(xs)
        elif dim == 4:
            return self.system.x4(xs)
        elif dim == 5:
            return self.system.x5(xs)
        else:
            raise AttributeError("generator can't deal with dim {}".format(dim))
