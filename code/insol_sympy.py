from sympy import pi, cos, sin, sqrt, atan, tan, symbols, Matrix
import numpy as np

#longitude, obliquity, latitude, precession, polar coord of earth
alpha,      beta,      phi,      rho,        theta = symbols('alpha beta phi rho theta')

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

sym_integral = alpha*sin(beta)*sin(phi)*cos(rho - theta) + (sin(alpha)*cos(beta)*cos(rho - theta) + sin(rho - theta)*cos(alpha))*cos(phi)

def calculate_daily_insol(theta_, rho_, beta_, phi_, point_on_circ):
    vals = {theta:theta_,rho:rho_, beta:beta_, phi:phi_}
    lim_end_x, lim_end_y = sym_lim_end[0].evalf(subs=vals), sym_lim_end[1].evalf(subs=vals)
    lim_start_x, lim_start_y = sym_lim_start[0].evalf(subs=vals), sym_lim_start[1].evalf(subs=vals)
    lim_end_z, lim_start_z = 2*[sin(phi_)]
    lim_end, lim_start = Matrix([lim_end_x, lim_end_y, lim_end_z]), Matrix([lim_start_x, lim_start_y, lim_start_z])
    if not lim_end_x.is_real:
        # This means the circle does not intersect with the day/night plane
        n = Matrix([cos(theta_), sin(theta_), 0])
        day = max(0,-n.dot(point_on_circ))
        if day:
            alpha_lim_start, alpha_lim_end = 0, 2*np.pi 
        else:
            return 0
    else:
        alpha_lim_end = np.arctan2(float(lim_end[1]),float(lim_end[0]))
        alpha_lim_start = np.arctan2(float(lim_start[1]),float(lim_start[0]))
    
    # Alter long vals to avoid integrating the wrong way round the earth 
    if alpha_lim_end < alpha_lim_start: alpha_lim_end+=2*np.pi
    if alpha_lim_end - alpha_lim_start > 2*np.pi: alpha_lim_end-=2*np.pi

    integral_subbed = sym_integral.subs(vals)
    integral_end = integral_subbed.subs(alpha, alpha_lim_end)
    integral_start = integral_subbed.subs(alpha, alpha_lim_start)

    return integral_end - integral_start


"""
#Here is the code used to generate the integral and lims above

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
"""
