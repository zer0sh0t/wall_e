import math
import wall_e

name = 'double_link_planar_robot'
type_ = 'rr' # 2 rotary joints
dh_params = [
     [math.pi/2, 0, 0, 20], # theta, d, alpha, a
     [math.pi, 0, 0, 10],
    ]
masses = [5, 7]
lengths = [20, 10]
dimensions = [[10], [10]] # [[r1], [r2]]
end_pos = [20, 20, 0] # qx, qy, qz
vels = [10, 5]
accs = [0, 0]
taus_req = [2207, 343]

r = wall_e.Robot(name, type_, dh_params, masses, lengths, dimensions)
print(r)

# render the robot
r.render()
 
# move the robot to the given position end position and render it
r.move(end_pos=end_pos) # you can also provide the final_angles instead of end_pos
 
# plot the trajectory for moving the robot to the given end position
r.move_traj(10, end_pos=end_pos) # you can also provide the final_angles instead of end_pos

# forward kinematics
fk_mat_expr, fk_mat = r.solve_fk()
wall_e.pprint(fk_mat_expr); print();
wall_e.pprint(fk_mat)

# inverse kinematics
thetas, fk_mat = r.solve_ik(end_pos)
print(thetas, end='\n'*2)
wall_e.pprint(fk_mat)

# inverse dynamics
tau_exprs, taus = r.solve_id(thetas, vels, accs)
for tau_expr, tau in zip(tau_exprs, taus):
    print('tau_expr:')
    wall_e.pprint(tau_expr)
    print(f'tau: {tau}', end='\n'*2)

# forward dynamics
'''
computation of forward dynamics takes a lot of time, i suggest you to create a new class and 
override the solve_id() method and rewrite the output tau expressions in numpy like shown below 
as sympy is super slow at numeric computation

class NewRobotClass(Robot):
    def solve_id():
        # write the inverse dynamics logic here in numpy
'''
optim_vals, taus = r.solve_fd(taus_req)
print(optim_vals)
print(taus)
