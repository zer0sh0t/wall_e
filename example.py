import math
import wall_e

dh_params = [
     [math.pi/2, 0, 0, 20], # theta, d, alpha, a
     [math.pi, 0, 0, 10],
    ]
end_pos = [20, 20, 0] # qx, qy, qz
vels = [10, 5]
accs = [0, 0]
masses = [5, 7]
dimensions = [[10], [10]] # [[r1], [r2]]
taus_req = [2207, 343]

bender = wall_e.Robot('bender', 'rr', dh_params) # 'rr' - 2 rotary joints
print(bender)

# forward kinematics
fk_mat_expr, fk_mat = bender.solve_fk()
wall_e.pprint(fk_mat_expr); print();
wall_e.pprint(fk_mat)

# inverse kinematics
thetas, fk_mat = bender.solve_ik(end_pos)
print(thetas, end='\n'*2)
wall_e.pprint(fk_mat)

# inverse dynamics
tau_exprs, taus = bender.solve_id(thetas, vels, accs, masses, dimensions)
for tau_expr, tau in zip(tau_exprs, taus):
    print('tau_expr:')
    wall_e.pprint(tau_expr)
    print(f'tau: {tau}', end='\n'*2)

# forward dynamics
'''
computation of forward dynamics takes a lot of time, i suggest you to create a new class and 
override the solve_id() method and rewrite the output tau expressions in numpy like shown below 
as sympy is super slow at numeric computation

class Bender(Robot):
    def solve_id():
        # write the inverse dynamics logic here in numpy
'''
optim_vals, taus = bender.solve_fd(masses, dimensions, taus_req)
print(optim_vals)
print(taus)
