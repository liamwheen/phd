from sympy import pi, cos, sin, sqrt, atan2, tan, symbols, Matrix, lambdify, Piecewise

#longitude, obliquity, latitude, precession, polar coord of earth
alpha,      beta,      phi,      rho,        theta = symbols('alpha beta phi rho theta')

# (x,y) coords for points over which to integrate (where there is daylight).
# These are in the fixed axes with earth considered upright (unrotated) but the
# sun (and so the day-night plane) is rotated inversely. 
sym_lim_end = ((-sqrt(cos(2*beta) + 2*cos(2*phi) - cos(2*rho - 2*theta) + cos(2*beta
    - 2*rho + 2*theta)/2 + cos(2*beta + 2*rho - 2*theta)/2 +
    1)*cos(beta)*cos(rho - theta)/(2*(sin(rho - theta)**2 +
        cos(beta)**2*cos(rho - theta)**2)) - sin(beta)*sin(phi)*sin(rho -
            theta)*cos(rho - theta)/(sin(beta)**2*cos(rho - theta)**2 -
                1))*tan(rho - theta)/cos(beta) - sin(phi)*tan(beta),
            -sqrt(cos(2*beta) + 2*cos(2*phi) - cos(2*rho - 2*theta) +
                cos(2*beta - 2*rho + 2*theta)/2 + cos(2*beta + 2*rho -
                    2*theta)/2 + 1)*cos(beta)*cos(rho - theta)/(2*(sin(rho -
                        theta)**2 + cos(beta)**2*cos(rho - theta)**2)) -
                    sin(beta)*sin(phi)*sin(rho - theta)*cos(rho -
                        theta)/(sin(beta)**2*cos(rho - theta)**2 - 1))

sym_lim_start = ((sqrt(cos(2*beta) + 2*cos(2*phi) - cos(2*rho - 2*theta) + cos(2*beta -
    2*rho + 2*theta)/2 + cos(2*beta + 2*rho - 2*theta)/2 + 1)*cos(beta)*cos(rho
        - theta)/(2*(sin(rho - theta)**2 + cos(beta)**2*cos(rho - theta)**2)) -
    sin(beta)*sin(phi)*sin(rho - theta)*cos(rho - theta)/(sin(beta)**2*cos(rho
        - theta)**2 - 1))*tan(rho - theta)/cos(beta) - sin(phi)*tan(beta),
    sqrt(cos(2*beta) + 2*cos(2*phi) - cos(2*rho - 2*theta) + cos(2*beta - 2*rho
        + 2*theta)/2 + cos(2*beta + 2*rho - 2*theta)/2 + 1)*cos(beta)*cos(rho -
            theta)/(2*(sin(rho - theta)**2 + cos(beta)**2*cos(rho - theta)**2))
        - sin(beta)*sin(phi)*sin(rho - theta)*cos(rho -
            theta)/(sin(beta)**2*cos(rho - theta)**2 - 1))

# Symbolic integral wrt longitude (alpha) of the equation for insol at specific
# point taken from "A Paleoclimate Model of Ice-Albedo..." - Richard McGehee et al
sym_integral = alpha*sin(beta)*sin(phi)*cos(rho - theta) + (sin(alpha)*cos(beta)*cos(rho - theta) + sin(rho - theta)*cos(alpha))*cos(phi)

# Condition such that if this is less than 0, the limits will be complex and the
# circle of constant latitude must lie entirely on one side of the day/night plane
real_condition = -sin(beta)**2*sin(phi)**2*cos(rho - theta)**2 + sin(rho -
        theta)**2*cos(phi)**2 + cos(beta)**2*cos(phi)**2*cos(rho - theta)**2

# Condition such that if 'real_condition' is < 0 i.e. limits are complex, then
# this will be negative if its entirely day and positive if its all night
day_night = cos(beta - phi)*cos(rho - theta)

# Turn x-y coords for limits into corresponding longitude
alpha_lim_start = atan2(*sym_lim_start[::-1])
alpha_lim_end = atan2(*sym_lim_end[::-1])
# Fix long limits so intetgration goes the right way round
# Should possible to do this better as it slows down simulation a lot
alpha_lim_end = Piecewise((alpha_lim_end - 2*pi, alpha_lim_end-alpha_lim_start>2*pi),
                          (alpha_lim_end + 2*pi, alpha_lim_end<alpha_lim_start),
                          (alpha_lim_end,True))

                          
                          

# Make piecewise limits since integral should be 0 if entirely night and over
# [0,2pi] if entirely day, otherwise calculated limits used.
lim_end = Piecewise((alpha_lim_end,real_condition>0),(0,day_night>0),(2*pi,day_night<=0))
lim_start = Piecewise((alpha_lim_start,real_condition>0),(0,True))

# Final integral given by I(end)-I(start) and turned into lambda function for use
final_integral = sym_integral.subs(alpha,lim_end) - sym_integral.subs(alpha,lim_start)
daily_insol_ratio = lambdify([rho,beta,theta,phi],final_integral/(2*pi),'sympy')


'''
#Here is the code used to generate the integral and cartesian lims above

from sympy import *
init_printing()
beta, rho, alpha = symbols('beta rho alpha')

# Find expression for insolation over an entire day at specific latitude
# Done by integrating over longitude
insol_latlon = cos(phi)*(cos(beta)*cos(theta - rho)*cos(alpha) + sin(theta -\
    rho)*sin(alpha)) + sin(phi)*sin(beta)*cos(theta - rho)

insol_lat = integrate(insol_latlon, alpha)

# Solve for intersection of circle of const latitude and day/night plane
x, y, z, phi, theta = symbols('x y z phi theta')

U_rho =  Matrix([[cos(rho), -sin(rho), 0],
                 [sin(rho), cos(rho) , 0],
                 [0       , 0        , 1]])

U_beta = Matrix([[cos(beta) , 0, sin(beta)],
                 [0         , 1, 0        ],
                 [-sin(beta), 0, cos(beta)]])

R = U_rho*U_beta
R_inv = R**-1

n = R_inv*(Matrix([cos(theta), sin(theta), 0]))

a, b, c = n

# Three constraints for x,y,z
plane = a*x + b*y + c*z
plane = trigsimp(plane.subs(z, sin(phi)))
circ = x**2 + y**2 - cos(phi)**2
sol = solve([plane, circ], [x, y], check=False)
'''
