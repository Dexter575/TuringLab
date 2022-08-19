#Step One:
# Import Dependencies
from math import cos
from math import e
from math import exp
from math import floor
from math import pi
from math import sin
from math import sqrt

def ackley(xy):
    '''
    Ackley Function

    wikipedia: https://en.wikipedia.org/wiki/Ackley_function

    global minium at f(x=0, y=0) = 0
    bounds: -5<=x,y<=5
    '''
    x, y = xy[0], xy[1]
    return (-20 * exp(-0.2 * sqrt(0.5 * (x*x + y*y))) -
            exp(0.5 * (cos(2.0*pi*x) + cos(2*pi*y))) + e + 20)

def cross_in_tray(xy):
    '''
    Cross-in-tray Fucntion

    global minimum:
        f(x=1.34941, y=1.34941) = -2.06261
        f(x=1.34941, y=-1.34941) = -2.06261
        f(x=-1.34941, y=1.34941) = -2.06261
        f(x=-1.34941, y=-1.34941) = -2.06261
    bounds: -10 = x, y <= 10
    '''
    x, y = xy[0], xy[1]
    return -0.0001*(abs(sin(x)*sin(y)*exp(abs(100-(sqrt(x**2 + y**2)/pi))))+1)**0.1

def eggholder(xy):
    '''
    Eggholder Function

    global minimum: f(x=512, y=404.2319) = -959.6407
    bounds: -512 <= x, y <= 512
    '''
    x, y = xy[0], xy[1]
    return (-(y+47)*sin(sqrt(abs((x/2.0) + (y+47)))) -
            x*sin(sqrt(abs(x-(y+47)))))

def rastrigin(x, safe_mode=False):
    '''Rastrigin Function
    References
    ----------
    wikipedia: https://en.wikipedia.org/wiki/Rastrigin_function
    '''

    if safe_mode:
        for item in x:
            assert x<=5.12 and x>=-5.12, 'input exceeds bounds of [-5.12, 5.12]'
    return len(x)*10.0 +  sum([item*item - 10.0*cos(2.0*pi*item) for item in x])

def sphere(x):
    '''
    Sphere Function

    global minimum at x=0 where f(x)=0
    bounds: none
    '''
    return sum([item * item for item in x])