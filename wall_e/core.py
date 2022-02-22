import math
import copy
import sympy as sp
from functools import partial
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation

pprint = lambda inp: sp.pprint(inp)

class Robot():
    def __init__(self, name, type_, dh_params, masses, lengths, dimensions):
        '''
        format of inputs:
        type_(str): l - linear, r - rotary
        dh_params(list[list]): [[theta_0, d_0, alpha_0, a_0], [theta_1, d_1, alpha_1, a_1], ...]
        masses(list): [mass_0, mass_1, ..., mass_n]
        lengths(list): [length_0, length_1, ..., length_n]
        dimensions(list[list]): [[radius_0], [radius_1], ..., [radius_n]] for joints of circular cross-sections
                              : [[a_0, b_0], [a_1, b_1], ..., [a_n, b_n]] for joints of rectangular cross-sections
                                where a is the length and b is the breadth of the rectangular cross-section
                              : [[radius_0], [a_1, b_1], [radius_2], ...] of course, it can be a combo of both
        '''
        self.name = name
        self.type = type_
        self.dh_params = dh_params
        self.masses = masses
        self.lengths = lengths
        self.dimensions = dimensions
        self.t = sp.symbols('t', real=True) # time
        self.clear_cache()

    # clear the cache if you want to call solve_fk() for the second with different set of dh_params
    def clear_cache(self):
        self.forward_params, self.forward_mats = [], []
        self.vel_params, self.acc_params = [], []
        self.tau_exprs = []

    def _screw_zx(self, i):
        if self.type[i] == 'r':
            theta = sp.Function(f'theta{i}')(self.t)
            vel = theta.diff(self.t)
            acc = vel.diff(self.t)
            d, alpha, a = sp.symbols(f'd{i} alpha{i} a{i}', real=True)
        elif self.type[i] == 'l':
            d = sp.Function(f'd{i}')(self.t)
            vel = d.diff(self.t)
            acc = vel.diff(self.t)
            theta, alpha, a = sp.symbols(f'theta{i} alpha{i} a{i}', real=True)

        ct, st, ca, sa = sp.cos(theta), sp.sin(theta), sp.cos(alpha), sp.sin(alpha)
        screw_mat = sp.Matrix([
                                [ct, -st*ca, sa*st, a*ct],
                                [st, ca*ct, -sa*ct, a*st],
                                [0, sa, ca, d],
                                [0, 0, 0, 1]
                            ])
        return screw_mat, theta, d, alpha, a, vel, acc

    # forward kinematics
    def solve_fk(self, dh_params=None):
        if dh_params == None:
            dh_params = copy.deepcopy(self.dh_params)

        if len(self.forward_mats) == 0:
            mat_expr = sp.eye(4)
            for i in range(len(dh_params)):
                screw_mat_expr, theta, d, alpha, a, vel, acc = self._screw_zx(i)
                mat_expr = mat_expr * screw_mat_expr
                self.forward_params.append([theta, d, alpha, a])
                self.vel_params.append(vel)
                self.acc_params.append(acc)
                self.forward_mats.append(mat_expr)

        subs_dict = {}
        for p, dhp in zip(self.forward_params, dh_params):
            for p_, dhp_ in zip(p, dhp):
                subs_dict[p_] = dhp_
        
        final_mat_expr = self.forward_mats[-1]
        final_mat = final_mat_expr.subs(subs_dict)
        return final_mat_expr, final_mat
    
    def _cost_fn_ik(self, initial_guess, end_pos, dh_params):
        j = 0
        for i in range(len(dh_params)):
            if self.type[i] == 'r':
                dh_params[i][0] = initial_guess[j]
                j += 1 
    
        fk_mat = self.solve_fk(dh_params)[1]
        qx, qy = float(fk_mat[0, 3]), float(fk_mat[1, 3])
        qx_req, qy_req, _ = end_pos
        cost = ((qx_req - qx) ** 2 + (qy_req - qy) ** 2) ** 0.5
        return cost

    # inverse kinematics
    def solve_ik(self, end_pos, tol=None):
        if self.type.count('l') == len(self.dh_params):
            raise ValueError('can\'t optimize joint angles of linear joints')
        else:
            dh_params = copy.deepcopy(self.dh_params)
            initial_guess = []
            for i in range(len(dh_params)):
                if self.type[i] == 'r':
                    initial_guess.append(dh_params[i][0])
            cost_fn = partial(self._cost_fn_ik, end_pos=end_pos, dh_params=dh_params)

            if tol == None:
                tol = 3e-4
            result = minimize(cost_fn, initial_guess, tol=tol)
            optim_thetas = result.x

            j = 0
            final_theta_vals = []
            for i in range(len(dh_params)):
                if self.type[i] == 'r':
                    theta = optim_thetas[j] 
                    dh_params[i][0] = theta
                    final_theta_vals.append(theta)
                    j += 1
                else:
                    final_theta_vals.append(dh_params[i][0])
            
            qz_req = end_pos[2]
            fk_mat = self.solve_fk(dh_params)[1]
            fk_mat[2, 3] = qz_req # just rotate the robot about z-axis to reach the given z-plane
            return final_theta_vals, fk_mat

    def _get_q_idx(self, i):
        if self.type[i] == 'r':
            q_idx = 0
        elif self.type[i] == 'l':
            q_idx = 1
        return q_idx

    def _U_ij(self, i, j):
        q_idx = self._get_q_idx(j)
        U_ij = self.forward_mats[i].diff(self.forward_params[j][q_idx])
        return U_ij

    def _U_ijk(self, i, j, k):
        U_ij = self._U_ij(i, j)
        q_idx = self._get_q_idx(k)
        U_ijk = U_ij.diff(self.forward_params[k][q_idx])
        return U_ijk

    def _D_ic(self, i, c, n, Js):
        D_ic = 0
        lower_bound = max(i, c)
        for j in range(lower_bound, n):
            U_jc = self._U_ij(j, c)
            J_j = Js[j]
            U_ji_T = self._U_ij(j, i).T
            D_ic += sp.Trace(U_jc * J_j * U_ji_T).rewrite(sp.Sum)
        return D_ic

    def _inertial_force(self, i, c, n, Js):
        D_ic = self._D_ic(i, c, n, Js)
        q_c_ddot = self.acc_params[c]
        inertial_force = D_ic * q_c_ddot
        return inertial_force

    def _h_icd(self, i, c, d, n, Js):
        h_icd = 0
        lower_bound = max(i, c, d)
        for j in range(lower_bound, n):
            U_jcd = self._U_ijk(j, c, d)
            J_j = Js[j]
            U_ji_T = self._U_ij(j, i).T
            h_icd += sp.Trace(U_jcd * J_j * U_ji_T).rewrite(sp.Sum)
        return h_icd

    def _coriolis_force(self, i, c, d, n, Js):
        h_icd = self._h_icd(i, c, d, n, Js)
        q_c_dot, q_d_dot = self.vel_params[c], self.vel_params[d]
        coriolis_force = h_icd * q_c_dot * q_d_dot
        return coriolis_force

    def _gravitational_force(self, i, n, masses, g, rs):
        gravitational_force = 0
        for j in range(i, n):
            m_j = masses[j]
            U_ji = self._U_ij(j, i)
            jj_r = rs[j]
            val = - m_j * g * U_ji * jj_r
            gravitational_force += val[0, 0]
        return gravitational_force
    
    # inverse dynamics
    def solve_id(self, thetas, vels, accs, ds=None):
        '''
        source: https://www.youtube.com/playlist?list=PLbRMhDVUMngcdUbBySzyzcPiFTYWr4rV_ lecture 24 to 29

        format of inputs:
        thetas(list): [theta_0, theta_1, ..., theta_n]
        vels(list): [vel_0, vel_1, ..., vel_n]
        accs(list): [acc_0, acc_1, ..., acc_n]
        lengths(list): [length_0, length_1, ..., length_n]

        formulation of inverse dynamics:
        n: number of joints
        J: interia tensor
        if rectangular_joint: m: mass of the joint, l: length of the joint, b: breadth of the joint
        if cicular_joint: m: mass of the joint, r: radius of the joint
        T: transformation matrix
        q: angle(if rotary) or offset(if linear)
        ii_r: position vector of ith frame wrt ith frame
        ii_v: velocity vector of ith frame wrt ith frame
        tau: torque(if rotary) or force(if linear)

        if rectangular_joint:
            J = [
                    [m*(a**2)/12, 0, 0, 0],
                    [0, m*(l**2)/3, 0, -m*l/2],
                    [0, 0, m*(b**2)/12, 0],
                    [0, -m*(l**2)/2, 0, m]
                ] 
        elif circular_joint:
            J = [
                    [m*(l**2)/3, 0, 0, -m*l/2],
                    [0, m*(r**2)/4, 0, 0],
                    [0, 0, m*(r**2)/4, 0],
                    [-m*l/2, 0, 0, m]
                ]

        g = [0, -9.81, 0, 0].T
        ii_r = [-L_i/2, 0, 0, 1]
        
        U_ij = d_i0_T / dq_j
        U_ijk = dU_ij / dq_k
        i0_v = sum_j_1_i(U_ij * q_j_dot) * ii_r # no need to compute this

        tau = sum_c_1_n(D_ic * q_c_ddot) + sum_c_1_n(sum_d_1_n(h_icd * q_c_dot * q_d_dot)) + C_i; i = 1 to n

        D_ic = sum_j_max(i, c)_n(Tr(U_jc * J_j * U_ji.T)); i, c = 1 to n # inertia term
        h_icd = sum_j_max(i, c, d)_n(Tr(U_jcd * J_j * U_ji.T)); i, c, d = 1 to n # coriolis and centrifugal term
        C_i = sum_j_i_n(- m_j * g * U_ji * jj_r); i = 1 to n # gravity term
        '''
        
        dh_params = copy.deepcopy(self.dh_params)
        lengths = copy.deepcopy(self.lengths)
        if ds == None:
            ds = [dhp[1] for dhp in self.dh_params]

        if len(self.tau_exprs) == 0:
            Js = []
            for m, l, d in zip(self.masses, lengths, self.dimensions):
                if len(d) == 1: # circular cross-section
                    r = d[0]
                    J = sp.Matrix([
                                    [m*(l**2)/3, 0, 0, -m*l/2],
                                    [0, m*(r**2)/4, 0, 0],
                                    [0, 0, m*(r**2)/4, 0],
                                    [-m*l/2, 0, 0, m]
                                ])
                elif len(d) == 2: # rectangular cross-section
                    a, b = d
                    J = sp.Matrix([
                                    [m*(a**2)/12, 0, 0, 0],
                                    [0, m*(l**2)/3, 0, -m*l/2],
                                    [0, 0, m*(b**2)/12, 0],
                                    [0, -m*(l**2)/2, 0, m]
                                ])
                Js.append(J)

            g = sp.Matrix([0, -9.81, 0, 0]).T
            rs = []
            for l in lengths:
                r = sp.Matrix([-l/2, 0, 0, 1])
                rs.append(r)

            n = len(dh_params)
            for i in range(n):
                inertial_force = 0
                coriolis_force = 0
                for c in range(n):
                    inertial_force += self._inertial_force(i, c, n, Js)
                    for d in range(n):
                        coriolis_force += self._coriolis_force(i, c, d, n, Js)

                gravitational_force = self._gravitational_force(i, n, self.masses, g, rs) # C_i
                tau_expr = inertial_force + coriolis_force + gravitational_force
                self.tau_exprs.append(tau_expr)
        
            '''
            order of substitution: a, alpha, acc, vel, theta, d
            values of higher order derivatives are substituted first and then the values of the lower ones,
            this is done to make sure that the lower order derivative values are not overwritten by the higher order ones
            to better understand this phenemenon, run this block of code:

            t = sp.symbols('t', real=True)
            theta_expr = sp.Function('theta')(t)
            vel_expr, acc_expr = theta_expr.diff(t), theta_expr.diff(t, 2)

            theta, vel, acc = 10, 20, 30
            eq = theta_expr + vel_expr + acc_expr # the expected value of this equation is 60
            value_0 = eq.subs(theta_expr, theta).subs(vel_expr, vel). subs(acc_expr, acc)
            value_1 = eq.subs(acc_expr, acc).subs(vel_expr, vel).subs(theta_expr, theta)
            print(f'value obtained by substituting in this order: theta, val, acc: {value_0}')
            print(f'value obtained by substituting in this order: acc, val, theta: {value_1}')
            '''

            alpha_a_dict = {}
            for p, dhp in zip(self.forward_params, dh_params):
                for i in range(len(p)):
                    if i == 0:
                        pass
                    elif i == 1:
                        pass
                    else:
                        alpha_a_dict[p[i]] = dhp[i]

            for i in range(len(self.tau_exprs)):
                tau_expr = self.tau_exprs[i]
                actual_tau_expr = tau_expr.subs(alpha_a_dict)
                self.tau_exprs[i] = actual_tau_expr

        acc_dict = {}
        vel_dict = {}
        theta_dict = {}
        d_dict = {}
        for p, t, ds in zip(self.forward_params, thetas, ds):
            for i in range(len(p)):
                if i == 0:
                    theta_dict[p[i]] = t
                elif i == 1:
                    d_dict[p[i]] = ds
                else:
                    pass
        for vp, v in zip(self.vel_params, vels):
            vel_dict[vp] = v
        for ap, a in zip(self.acc_params, accs):
            acc_dict[ap] = a

        taus = []
        for tau_expr in self.tau_exprs:
            tau = tau_expr.subs(acc_dict).subs(vel_dict).subs(theta_dict).subs(d_dict)
            taus.append(tau)
        return self.tau_exprs, taus

    def _get_vals(self, list_):
        n = len(self.dh_params)
        thetas, ds, vels, accs = [], [], [], []
        j = 0
        for i in range(len(list_)):
            val = list_[i]
            if 0 <= i < n:
                if self.type[j] == 'r':
                    thetas.append(val) # this value will be optimized
                    ds.append(self.dh_params[j][1]) # whereas this value won't be optimized as this joint can't change it's length ##################
                elif self.type[j] == 'l':
                    ds.append(val) # this value will be optimized
                    thetas.append(self.dh_params[j][0]) # whereas this value won't be optimized as this joint can't change its angle
                j += 1
            elif n <= i < 2*n:
                vels.append(val)
            else:
                accs.append(val)
        return thetas, ds, vels, accs

    def _cost_fn_fd(self, initial_guess, taus_req):
        thetas, ds, vels, accs = self._get_vals(initial_guess)
        taus = self.solve_id(thetas, vels, accs, ds)[1]
        cost = 0
        for tr, t in zip(taus_req, taus):
            cost += (tr - t) ** 2
        cost = cost ** 0.5
        return cost

    # forward dynamics
    def solve_fd(self, taus_req, tol=None):
        initial_guess = [0 for _ in range(3*len(self.dh_params))]
        cost_fn = partial(self._cost_fn_fd, taus_req=taus_req)
        if tol == None:
            tol = 3e-4

        result = minimize(cost_fn, initial_guess, tol=tol)
        optim_vals = result.x
        thetas, ds, vels, accs = self._get_vals(optim_vals)
        taus = self.solve_id(thetas, vels, accs, ds)[1]
        optim_vals = {'thetas': thetas, 'offsets(d)': ds, 'vels': vels, 'accs': accs}
        return optim_vals, taus
        
    def _get_pts(self, s_pt, length, angle, d, init_z=0):
        x, y, z = s_pt
        end_x = x + length * math.cos(angle)
        end_y = y + length * math.sin(angle)
        end_z = z + d + init_z
        offset_pts = (end_x, end_y, z + init_z)
        end_pts = (end_x, end_y, end_z)
        return offset_pts, end_pts

    def _render(self, angles, ds, init_z, reinit):
        if angles == None:
            angles = [dhp[0] for dhp in self.dh_params]
        if ds == None:
            ds = [dhp[1] for dhp in self.dh_params]
        lengths = copy.deepcopy(self.lengths)
        
        if reinit:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = self.fig
            ax = self.ax
            ax.clear()
        ax.set_title(f'{self.name}')

        s_pt = (0, 0, 0)
        for i in range(len(angles)):
            if i != 0:
                angle = angles[i] + angles[i-1]
                iz = 0
                c = 'r'
            else:
                angle = angles[i]
                iz = init_z
                c = 'm' # base
            
            o_pt, e_pt = self._get_pts(s_pt, lengths[i], angle, ds[i], iz)
            ax.scatter(s_pt[2], s_pt[1], zs=s_pt[0], c=c) # current point
            ax.scatter(o_pt[2], o_pt[1], zs=o_pt[0], c='r') # offset point

            # s_pt -> o_pt -> e_pt
            ax.plot([s_pt[2], o_pt[2]], [s_pt[1], o_pt[1]], zs=[s_pt[0], o_pt[0]], c='b')
            ax.plot([o_pt[2], e_pt[2]], [o_pt[1], e_pt[1]], zs=[o_pt[0], e_pt[0]], c='b')
            s_pt = e_pt
        ax.scatter(s_pt[2], s_pt[1], zs=s_pt[0], c='g') # end effector

    # render the robot at home position provided in the dh paramters
    def render(self, angles=None, ds=None, init_z=0):
        self._render(angles, ds, init_z, True)
        plt.show()

    def move(self, end_pos=None, thetas=None, ds=None, ret=False):
        '''
        moves the robot to the given position/angle
        provide either end_pos or thetas

        end_pos(tuple): (x_final, y_final, z_final)
        thetas(list): [theta_0, theta_1, ...]
        optional:
        ds(list): [d_0, d_1, ...]
        '''
        if len(self.forward_mats) == 0:
            _, _ = self.solve_fk() # collect cache

        if end_pos != None: # get angles using inveres kinematics
            thetas, _ = self.solve_ik(end_pos)
        elif thetas != None:
            pass # we've got everything we need
        else:
            raise ValueError('please specify either the end position or the joint angles!!')

        if ds == None:
            home_z = sum([dhp[1] for dhp in self.dh_params])
        else:
            home_z = sum(ds)

        # if end_pos[2] > home_z:
        #     init_z = end_pos[2] - home_z
        # else:
        #     init_z = - (abs(end_pos[2]) - home_z)
        init_z = end_pos[2] - home_z

        if ret:
            return thetas, init_z
        else:
            self.render(angles=thetas, ds=ds, init_z=init_z)

    def _cubic_fn(self, q_i, q_f, ts, t):
        val = q_i + (3 * (q_f - q_i) / (ts**2)) * (t**2) - (2 * (q_f - q_i) / (ts**3)) * (t**3)
        return val

    def _cubic_dot_fn(self, q_i, q_f, vel_i, vel_f, ts, t):
        val = q_i + vel_i * t + ((3 * (q_f - q_i) / (ts**2)) - (2 * vel_i / ts) - (vel_f / ts)) * (t**2) + \
              ((- 2 * (q_f - q_i) / (ts**3)) + ((vel_f + vel_i) / (ts**2))) * (t**3)
        return val

    def _fifth_fn(self, th_i, th_f, vel_i, vel_f, acc_i, acc_f, ts, t):
        val = th_i + vel_i * t + (acc_i * (t**2) / 2) + ((20 * (th_f - th_i) - (8 * vel_f + 12 * vel_i) * ts - \
              (3 * acc_i - acc_f) * (ts**2)) / 2 * (ts**3)) * (t**3) + ((30 * (th_i - th_f) + (14 * vel_f + 16 * vel_i) * ts + \
              (3 * acc_i - 2 * acc_f) * (ts**2)) / 2 * (ts**4)) * (t**4)+ ((12 * (th_f - th_i) - 6 * (vel_f + vel_i) * ts - \
              (acc_i - acc_f) * (ts**2)) / 2 * (ts**5)) * (t**5) 
        return val

    def _create_anim(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def _anim_fn(self, i, thetas_across_time, ds_across_time, init_z):
        z = self._cubic_fn(0, init_z, len(thetas_across_time)-1, i) # smooth translation towards the end z plane
        self._render(angles=thetas_across_time[i], ds=ds_across_time[i], init_z=z, reinit=False) 

    def move_traj(self, ts, end_pos=None, final_angles=None, final_ds=None, fn='cubic', final_vels=None, final_accs=None):
        '''
        plots the trajectory of the robot
        provide either end_pos or final_angles

        ts(int): time taken by the robot to reach end position/final angles
        end_pos(tuple): (x_final, y_final, z_final)
        final_angles(list): [theta_0, theta_1, ...]

        fn(str): 'cubic', 'cubic_dot' or 'fifth'
        'cubic' - use this if final_vels == None
        'cubic_dot' - use this if final_vels != None but final_accs == None
        'fifth' - use this if final_vels != None and final_accs != None

        optional:
        final_ds(list): [d_0, d_1, ...] # for linear joints
        final_vels(list): [vel_0, vel_1, ...]
        final_accs(list): [acc_0, acc_1, ...]
        '''
        if fn not in ['cubic', 'cubic_dot', 'fifth']:
            raise ValueError('please input a valid trajectory function!!')

        init_angles = [dhp[0] for dhp in self.dh_params]
        init_ds = [dhp[1] for dhp in self.dh_params]
        if end_pos != None:
            final_angles, init_z = self.move(end_pos, ret=True)
        
        if fn != 'cubic':
            vel_i = 0
        if fn == 'fifth':
            acc_i = 0
        
        thetas_across_time = []
        ds_across_time = []
        for t in range(ts):
            thetas = []
            ds = []

            for i in range(len(self.dh_params)):
                if self.type[i] == 'r':
                    q_i, q_f = init_angles[i], final_angles[i]
                elif self.type[i] == 'l':
                    q_i = init_ds[i]
                    if final_ds != None:
                        q_f = final_ds[i]
                    else:
                        q_f = q_i

                if fn != 'cubic':
                    vel_f = final_vels[i]
                if fn == 'fifth':
                    acc_f = final_accs[i]

                if q_f != q_i: # happens only when the linear joint's offset is not changed across time
                    if fn == 'cubic':
                        val = self._cubic_fn(q_i, q_f, ts-1, t)
                    elif fn == 'cubic_dot':
                        val = self._cubic_dot_fn(q_i, q_f, vel_i, vel_f, ts-1, t)
                    elif fn == 'fifth':
                        val = self._fifth_fn(q_i, q_f, vel_i, vel_f, acc_i, acc_f, ts-1, t)
                else:
                    val = q_i
                
                if self.type[i] == 'r':
                    thetas.append(val)
                    ds.append(self.dh_params[i][1])
                elif self.type[i] == 'l':
                    ds.append(val)
                    thetas.append(self.dh_params[i][0])

            thetas_across_time.append(thetas)
            ds_across_time.append(ds)

        self._create_anim()
        anim_fn = partial(self._anim_fn, thetas_across_time=thetas_across_time, ds_across_time=ds_across_time, init_z=init_z)
        anim = FuncAnimation(self.fig, anim_fn, frames=ts, repeat=False)
        plt.show()

    def __repr__(self):
        return f'Robot(name={self.name}, type={self.type})'
