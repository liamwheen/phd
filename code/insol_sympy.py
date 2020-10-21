from sympy import cos, sin, sqrt, tan, symbols

#longitude, obliquity, latitude, precession, solar angle
alpha,      beta,      phi,      rho,        theta = symbols('alpha beta phi rho theta')

lim_a = ((-sqrt(cos(2*beta) + 2*cos(2*phi) - cos(2*rho - 2*theta) + cos(2*beta
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

lim_b = ((sqrt(cos(2*beta) + 2*cos(2*phi) - cos(2*rho - 2*theta) + cos(2*beta -
    2*rho + 2*theta)/2 + cos(2*beta + 2*rho - 2*theta)/2 + 1)*cos(beta)*cos(rho
        - theta)/(2*(sin(rho - theta)**2 + cos(beta)**2*cos(rho - theta)**2)) -
    sin(beta)*sin(phi)*sin(rho - theta)*cos(rho - theta)/(sin(beta)**2*cos(rho
        - theta)**2 - 1))*tan(rho - theta)/cos(beta) - sin(phi)*tan(beta),
    sqrt(cos(2*beta) + 2*cos(2*phi) - cos(2*rho - 2*theta) + cos(2*beta - 2*rho
        + 2*theta)/2 + cos(2*beta + 2*rho - 2*theta)/2 + 1)*cos(beta)*cos(rho -
            theta)/(2*(sin(rho - theta)**2 + cos(beta)**2*cos(rho - theta)**2))
        - sin(beta)*sin(phi)*sin(rho - theta)*cos(rho -
            theta)/(sin(beta)**2*cos(rho - theta)**2 - 1))

integral = alpha*sin(beta)*sin(phi)*cos(rho - theta) + (sin(alpha)*cos(beta)*cos(rho - theta) + sin(rho - theta)*cos(alpha))*cos(phi)

def calculate_daily_insol(theta_, rho_, beta_, phi_):
    vals = {theta:theta_,rho:rho_, beta:beta_, phi:phi_}
    lim_a_x, lim_a_y = lim_a[0].evalf(subs=vals), lim_a[1].evalf(subs=vals)
    lim_b_x, lim_b_y = lim_b[0].evalf(subs=vals), lim_b[1].evalf(subs=vals)
    lim_a_z, lim_b_z = 2*[sin(phi_)]
    lim_a, lim_b = [lim_a_x, lim_a_y, lim_a_z], [lim_b_x, lim_b_y, lim_b_z]
    if isinstance(lim_a_x,sympy.core.add.Add):
        print('ITS COMPLEX')
    ---------finish this to check that the circle intersects or doest (if
            complex) in which case maybe do similar check to dot product theta
    and rotate latlon=0 vec for pos/neg value if its day/night
    ---------get the lims into the form of longitude (may need to rotation
            matrix the limit vectors) then use 'integral' and sub in the values
            of alpha

#def integral(theta_, rho_, beta_, phi_, lims):


#Here is the code used to generate the integral and lims above


"""
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
