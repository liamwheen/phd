from numpy import pi, cos, sin, sqrt, arctan2, tan, greater, less, select, nan, logical_and
from scipy.integrate import quad_vec
# This is done to prevent the warning that appears when the sqrt of a negative
# is taken. This complex value is never used (due to the piecewise condition)
# but is still calculated before being checked.
#import warnings
#warnings.filterwarnings("ignore")

def calc_daily_average(rho, beta, theta, phi, eps):
    x0 = rho - theta
    x1 = sin(x0)
    x2 = sin(beta)
    x3 = cos(x0)
    x4 = x3**2
    x5 = x2**2*x4
    x6 = sin(phi)
    x7 = x2*x3*x6
    x8 = -x1*x7/(x5 - 1)
    x9 = x1**2
    x10 = cos(beta)
    x11 = x10**2*x4
    x12 = 2*beta
    x13 = 2*rho
    x14 = 2*theta
    x15 = x13 - x14
    x16 = x10*x3
    x17 = x16*sqrt(abs(2*cos(2*phi) + cos(x12) - cos(x15) + cos(x12 + x15)/2 +
        cos(x12 - x13 + x14)/2 + 1))/(2*(x11 + x9))
    x18 = x17 + x8
    x19 = -x6*tan(beta)
    x20 = tan(x0)/x10
    x21 = arctan2(x18, x18*x20 + x19)
    x22 = cos(phi)
    x23 = x22**2
    x24 = x11*x23 + x23*x9 - x5*x6**2 > 0
    x25 = select([x24,True], [x21,0], default=nan)
    x26 = 2*pi
    x27 = -x17 + x8
    x28 = arctan2(x27, x19 + x20*x27)
    x29 = select([x24,greater(x3*cos(beta - phi), 0),True], [select([less(x26,
        -x21 + x28),greater(x21, x28),True], [-x26 + x28,x26 + x28,x28],
        default=nan),0,x26], default=nan)
    return 1.52795762923834e+25*(-eps*cos(theta) + 1)**2*(-x22*(-x1*cos(x25) -
        x16*sin(x25)) + x22*(-x1*cos(x29) - x16*sin(x29)) + x25*x7 -
        x29*x7)/(pi*(149597870700 - 149597870700*eps**2)**2)


def calc_yearly_average(beta, phi, eps):
    def g(theta):
        x0 = 1/pi
        x1 = 2*pi
        x2 = theta - 1
        x3 = sin(x2)
        x4 = sin(beta)
        x5 = cos(x2)
        x6 = x5**2
        x7 = x4**2*x6
        x8 = sin(phi)
        x9 = x4*x5*x8
        x10 = x3*x9/(x7 - 1)
        x11 = x3**2
        x12 = cos(beta)
        x13 = x12**2*x6
        x14 = 2*beta
        x15 = 2*theta
        x16 = x15 - 2
        x17 = x12*x5
        x18 = x17*sqrt(abs(2*cos(2*phi) + cos(x14) - cos(x16) + cos(x14 +
            x16)/2 + cos(x14 - x15 + 2)/2 + 1))/(2*(x11 + x13))
        x19 = x10 - x18
        x20 = -x8*tan(beta)
        x21 = tan(x2)/x12
        x22 = -x19*x21 + x20
        x23 = arctan2(x19, x22)
        x24 = -x9*(-x1 + x23)
        x25 = cos(phi)
        x26 = 1/sqrt(x19**2 + x22**2)
        x27 = x10 + x18
        x28 = x20 - x21*x27
        x29 = 1/sqrt(x27**2 + x28**2)
        x30 = arctan2(x27, x28)
        x31 = -x25*(-x17*x27*x29 + x28*x29*x3) + x30*x9
        x32 = x25*(-x17*x19*x26 + x22*x26*x3) + x31
        x33 = 1.52795762923834e+25*x0
        x34 = x25**2
        x35 = x11*x34 + x13*x34 - x7*x8**2 > 0
        x36 = x23 - x30 > x1
        x37 = x23 < x30
        x38 = x35 | x37
        x39 = x35 | x36
        x40 = x35 | x36 | x37
        x41 = -x9*(x1 + x23)
        x42 = -x23*x9
        x43 = x5*cos(beta - phi) > 0
        x44 = x25*x3 + x31
        return x0*select([logical_and.reduce((x35,x36,x38,x39,x40)),
        logical_and.reduce((x35,x37,x38,x39,x40)),
        logical_and.reduce((x35,x38,x39,x40)),
        logical_and.reduce((x35,x38,x39,x40,x43)),
        logical_and.reduce((x35,x36)),
        logical_and.reduce((x35,x37)),x35,x43,True],
        [x33*(x24 + x32),x33*(x32 + x41),x33*(x32 + x42),x32*x33,x33*(x24 +
        x44),x33*(x41 + x44),x33*(x42 + x44),0,-3.05591525847668e+25*x9],
        default=nan)/(44759045835947836980000*sqrt(1 - eps**2))
    return quad_vec(g, 0, 2*pi)[0]


"""
# Here is the code used to develop the above function

from sympy import symbols, sin, cos, atan2,  pi, sqrt, solve, trigsimp, simplify
from sympy import Piecewise, Float, Integral, integrate, lambdify, Matrix, trigsimp

#obliq, precess, long,  eccen
beta,   rho,     gamma, eps   = symbols('beta rho gamma eps')
#cartesians, lat, polar angle of earth
x, y, z,     phi, theta                = symbols('x y z phi theta')
a = 149597870700 # Semi-major axis (1au)
q = Float(3.055915258476678e+25) # Scaled irradiance constant

# Precession rotation matrix
U_rho =  Matrix([[cos(rho), -sin(rho), 0],
                 [sin(rho), cos(rho) , 0],
                 [0       , 0        , 1]])
# Obliquity rotation matrix
U_beta = Matrix([[cos(beta) , 0, sin(beta)],
                 [0         , 1, 0        ],
                 [-sin(beta), 0, cos(beta)]])

R = U_rho*U_beta
R_inv = trigsimp(R**-1)

# Assemble "day-night plane" and "circle of const lat" equations on stationary earth axes
# Rotate n (sun to earth vec which is normal to the day-night plane) 
# with R^-1 so the sun moves around an unrotated earth
i,j,k = R_inv*(Matrix([cos(theta), sin(theta), 0]))
# cos(phi) is the radius of the circle of const lat
circ = x**2 + y**2 - cos(phi)**2
# Plane always crosses through earth origin so no constants in plane equation
plane = i*x + j*y + k*z
# z will equal sin(latitude) when the plane intersects the cirle of that latitude
plane = trigsimp(plane.subs(z, sin(phi)))
# Find x-y coords of where plane intersects circle (may be two points or none)
sol = solve([plane, circ], [x, y], check=False)
cart_end, cart_start = sol

# Convert the x-y coords into longitude angle
gamma_start = atan2(*cart_start[::-1])
gamma_end = atan2(*cart_end[::-1])
# Introduce conditions so that integration goes round the right way and only once
gamma_end = Piecewise((gamma_end - 2*pi, gamma_end-gamma_start>2*pi),
                      (gamma_end + 2*pi, gamma_end<gamma_start),
                      (gamma_end, True))

# Express r in terms of orbital parameters (so it's not explicity a function of time)
r_eps = a*(1-eps**2)/(1-eps*cos(theta))
r = symbols('r')

# Insolation at a specific lat and lon given orbital params - "A paleoclimate model..." McGehee '12
insol_latlon = (sin(gamma)*sin(rho - theta) - cos(beta)*cos(gamma)*cos(rho -
    theta))*cos(phi) - sin(beta)*sin(phi)*cos(rho - theta)

# Determine whether circle intersects the day-night plane
intersects_plane = -sin(beta)**2*sin(phi)**2*cos(rho - theta)**2 + sin(rho -
        theta)**2*cos(phi)**2 + cos(beta)**2*cos(phi)**2*cos(rho - theta)**2 > 0
# If no intersection - determine if it's entirely day or entirely night
all_day = cos(beta - phi)*cos(rho - theta) <= 0

# Collect everything into conditional limits for integrating over longitude
start = Piecewise((gamma_start,intersects_plane),(0,True))
end = Piecewise((gamma_end,intersects_plane),(0,~all_day),(2*pi,all_day))

# Integrate over longitude from start to end and divide by 2pi to give average
# insol over 24 hours (treating earth as static in this time)
daily_average = q/r**2*integrate(insol_latlon, (gamma,start,end))/(2*pi)
# Turn integral into lambda function for quick substitution
calc_daily_average = lambdify([rho, beta, theta, phi, eps],
        daily_average.subs(r,r_eps),'numpy')

# Integrate this daily average over a year (theta [0,2pi])
# To change integration variable from t to theta, we must multiply by dt/dtheta
# which is P*r^2/(2pi*a*b) where P is period but can be cancelled since we'll
# divide by P to get the average insol over the year (not the total)
dtheta_coef = 1/(2*pi*a*a*sqrt(1-eps**2))
yearly_average = dtheta_coef*Integral(daily_average*r**2, (theta,0,2*pi))
calc_yearly_average = lambdify([rho, beta, phi, eps],
        yearly_average,'sympy')

# Then use sympy.cse to get the decomposition of yearly_average for
# much faster substitution

"""
