from sympy import *
init_printing()
beta = Symbol('beta',positive=True)
rho, alpha, phi, theta = symbols('rho alpha phi theta')
r, h = cos(phi), sin(phi)

U_rho =  Matrix([[cos(rho), -sin(rho), 0],
                 [sin(rho), cos(rho) , 0],
                 [0       , 0        , 1]])
U_beta = Matrix([[cos(beta) , 0, sin(beta)],
                 [0         , 1, 0        ],
                 [-sin(beta), 0, cos(beta)]])

R = U_rho*U_beta

circ = R*(Matrix([0,0,h]) + cos(alpha)*Matrix([r,0,0]) + sin(alpha)*Matrix([0,r,0]))

n = Matrix([cos(theta), sin(theta), 0])

eq = trigsimp(n.dot(circ))
pprint(eq)
x,y = symbols('x y')
# Solve for cos,sin of alpha instead to lighten the load (takes ~2 mins)
pprint(solve([eq.subs(sin(alpha),x).subs(cos(alpha),y), x**2 + y**2-1], [x,y]))
