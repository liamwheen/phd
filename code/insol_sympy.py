from numpy import pi, cos, sin, sqrt, arctan2, tan, greater, less, select, nan, logical_and
from scipy.integrate import quad_vec
# This is done to prevent the warning that appears when the sqrt of a negative
# is taken. This complex value is never used (due to the piecewise condition)
# but is still calculated before being checked.
import warnings
warnings.filterwarnings("ignore")

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
    x17 = x16*sqrt(2*cos(2*phi) + cos(x12) - cos(x15) + cos(x12 + x15)/2 + cos(x12 - x13 + x14)/2 + 1)/(2*x11 + 2*x9)
    x18 = x17 + x8
    x19 = -x6*tan(beta)
    x20 = tan(x0)/x10
    x21 = arctan2(x18, x18*x20 + x19)
    x22 = cos(phi)
    x23 = x22**2
    x24 = x11*x23 + x23*x9 - x5*x6**2 > 0
    x25 = select([x24,True], [x21,0], default=nan)
    x26 = x1*x22
    x27 = x16*x22
    x28 = 1365.49615900093*(eps*cos(theta) - 1)**2/(eps**2 - 1)**2
    x29 = 2*pi
    x30 = -x17 + x8
    x31 = arctan2(x30, x19 + x20*x30)
    x32 = select([x24,greater(x3*cos(beta - phi), 0),True], 
            [select([less(x29, -x21 + x31),greater(x21, x31),True], 
            [-x29 + x31,x29 + x31,x31], default=nan),0,x29], default=nan)
    return (x28*(x25*x7 + x26*cos(x25) + x27*sin(x25)) - x28*(x26*cos(x32) + x27*sin(x32) + x32*x7))/(2*pi)


def calc_yearly_average(rho, beta, phi, eps):
    def g(theta):
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
        x17 = x16*sqrt(2*cos(2*phi) + cos(x12) - cos(x15) + cos(x12 + x15)/2 + cos(x12 - x13 + x14)/2 + 1)/(2*(x11 + x9))
        x18 = x17 + x8
        x19 = -x6*tan(beta)
        x20 = tan(x0)/x10
        x21 = x18*x20 + x19
        x22 = arctan2(x18, x21)
        x23 = 1/sqrt(x18**2 + x21**2)
        x24 = cos(phi)
        x25 = x16*x24
        x26 = x1*x24
        x27 = eps**2
        x28 = (eps*cos(theta) - 1)**2
        x29 = 1365.49615900093*x28/(x27 - 1)**2
        x30 = x29*(x18*x23*x25 + x21*x23*x26 + x22*x7)
        x31 = 2*pi
        x32 = -x17 + x8
        x33 = x19 + x20*x32
        x34 = arctan2(x32, x33)
        x35 = x7*(-x31 + x34)
        x36 = 1/sqrt(x32**2 + x33**2)
        x37 = x25*x32*x36 + x26*x33*x36
        x38 = (1 - x27)**(3/2)/(4*pi**2*x28)
        x39 = x24**2
        x40 = x11*x39 + x39*x9 - x5*x6**2 > 0
        x41 = -x22 + x34 > x31
        x42 = x34 < x22
        x43 = x40 | x42
        x44 = x40 | x41
        x45 = x40 | x41 | x42
        x46 = x7*(x31 + x34)
        x47 = x34*x7
        x48 = x3*cos(beta - phi) > 0
        return select([
            logical_and.reduce((x40, x41, x43, x44, x45)),
            logical_and.reduce((x40, x42, x43, x44, x45)),
            logical_and.reduce((x40, x43, x44, x45)),
            logical_and.reduce((x40, x43, x44, x45, x48)),
            logical_and.reduce((x40, x41)),
            logical_and.reduce((x40, x42)),x40,x48,True],
            [x38*(-x29*(x35 + x37) + x30),
            x38*(-x29*(x37 + x46) + x30),
            x38*(-x29*(x37 + x47) + x30),
            x38*(-x29*x37 + x30),
            x38*(-x29*(x26 + x35) + x30),
            x38*(-x29*(x26 + x46) + x30),
            x38*(-x29*(x26 + x47) + x30), 0,
            x38*(x26*x29 - x29*(x26 + x31*x7))],default=nan)
    return quad_vec(g, 0, 2*pi)[0]


"""
# Here is the code used to develop the above function

from sympy import symbols, sin, cos, atan2,  pi, sqrt, solve, trigsimp, simplify
from sympy import Piecewise, Float, Integral, integrate, lambdify, Matrix

#obliq, precess, long,  eccen
beta,   rho,     alpha, eps   = symbols('beta rho alpha eps')
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
R_inv = R**-1

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
alpha_start = atan2(*cart_start[::-1])
alpha_end = atan2(*cart_end[::-1])
# Introduce conditions so that integration goes round the right way and only once
alpha_end = Piecewise((alpha_end - 2*pi, alpha_end-alpha_start>2*pi),
                      (alpha_end + 2*pi, alpha_end<alpha_start),
                      (alpha_end, True))

# Express r in terms of orbital parameters (so it's not explicity a function of time)
r = a*(1-eps**2)/(1-eps*cos(theta))
insol = simplify(q/r**2)

# Insolation at a specific lat and lon given orbital params - "A paleoclimate model..." McGehee '12
insol_latlon = -insol*(cos(phi)*(cos(beta)*cos(theta - rho)*cos(alpha) + sin(theta -\
    rho)*sin(alpha)) + sin(phi)*sin(beta)*cos(theta - rho))

# Determine whether circle intersects the day-night plane
intersects_plane = -sin(beta)**2*sin(phi)**2*cos(rho - theta)**2 + sin(rho -
        theta)**2*cos(phi)**2 + cos(beta)**2*cos(phi)**2*cos(rho - theta)**2 > 0
# If no intersection - determine if it's entirely day or entirely night
all_day = cos(beta - phi)*cos(rho - theta) <= 0

# Collect everything into conditional limits for integrating over longitude
start = Piecewise((alpha_start,intersects_plane),(0,True))
end = Piecewise((alpha_end,intersects_plane),(0,~all_day),(2*pi,all_day))

# Integrate over longitude from start to end and divide by 2pi to give average
# insol over 24 hours (treating earth as static in this time)
daily_average = integrate(insol_latlon, (alpha,start,end))/(2*pi)
# Turn integral into lambda function for quick substitution
calc_daily_average = lambdify([rho, beta, theta, phi, eps],
        daily_average,'numpy')

# Integrate this daily average over a year (theta [0,2pi])
# To change integration variable from t to theta, we must multiply by dt/dtheta
# which is P*r^2/(2pi*a*b) where P is period but can be cancelled since we'll
# divide by P to get the average insol over the year (not the total)
dtheta_coef = simplify(r**2/(2*pi*a*a*sqrt(1-eps**2)))
yearly_average = Integral(daily_average*dtheta_coef, (theta,0,2*pi))
calc_yearly_average = lambdify([rho, beta, phi, eps],
        yearly_average,'sympy')

# Then use sympy.cse to get the decomposition of the yearly lambda function for
# much faster substitution
"""

