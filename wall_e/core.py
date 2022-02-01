import math
import copy
import sympy as sp
from functools import partial
from scipy.optimize import minimize

def pprint(inp):
    sp.pprint(inp)

class Robot():
    def __init__(self, name, type_, dh_params):
        '''
        format of inputs:
        type_(str): l - linear, r - rotary
        dh_params(list): theta, d, alpha, a
        '''
        self.name = name
        self.type = type_
        self.dh_params = dh_params
        self.t = sp.symbols('t', real=True)
        self.clear_cache()

    # clear the cache before you call solve_fk() second time with different dh_params than the ones initially provided
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
            a = sp.Function(f'a{i}')(self.t)
            vel = a.diff(self.t)
            acc = vel.diff(self.t)
            theta, d, alpha = sp.symbols(f'theta{i} d{i} alpha{i}', real=True)

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
        qx, qy, qz = float(fk_mat[0, 3]), float(fk_mat[1, 3]), float(fk_mat[2, 3])
        qx_req, qy_req, qz_req = end_pos
        cost = ((qx_req - qx) ** 2 + (qy_req - qy) ** 2 + (qz_req - qz) ** 2) ** 0.5
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

            fk_mat = self.solve_fk(dh_params)[1]
            return final_theta_vals, fk_mat

    def _get_q_idx(self, i):
        if self.type[i] == 'r':
            q_idx = 0
        elif self.type[i] == 'l':
            q_idx = 3
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
    def solve_id(self, thetas, vels, accs, masses, dimensions, lengths=None):
        '''
        source: https://www.youtube.com/playlist?list=PLbRMhDVUMngcdUbBySzyzcPiFTYWr4rV_ lecture 24 to 29

        format of inputs:
        thetas = [theta_0, theta_1, ..., theta_n]
        vels = [vel_0, vel_1, ..., vel_n]
        accs = [acc_0, acc_1, ..., acc_n]
        masses = [mass_0, mass_1, ..., mass_n]
        dimensions = [[radius_0], [radius_1], ..., [radius_n]] for joints of circular cross-sections
                   = [[a_0, b_0], [a_1, b_1], ..., [a_n, b_n]] for joints of rectangular cross-sections
                     where a is the length and b is the breadth of the rectangular cross-section
                   = [[radius_0], [a_1, b_1], [radius_2], ...] of course, it can be a combo of both
        lengths = [length_0, length_1, ..., length_n]

        formulation of inverse dynamics:
        n: number of joints
        J: interia tensor
        if rectangular_joint: m: mass of the joint, l: length of the joint, b: breadth of the joint
        if cicular_joint: m: mass of the joint, r: radius of the joint
        T: transformation matrix
        q: angle(if rotary) or length(if linear)
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
        if lengths == None:
            lengths = [dhp[3] for dhp in dh_params]

        if len(self.tau_exprs) == 0:
            Js = []
            for m, l, d in zip(masses, lengths, dimensions):
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

                gravitational_force = self._gravitational_force(i, n, masses, g, rs) # C_i
                tau_expr = inertial_force + coriolis_force + gravitational_force
                self.tau_exprs.append(tau_expr)
        
        '''
        order of substitution: d, alpha, acc, vel, theta, a
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

        d_alpha_dict = {} # wanted to keep a cache of this dict as it contains static values but sympy outputs unsolved expressions by using cached values, weird behavior
        acc_dict = {}
        vel_dict = {}
        theta_dict = {}
        a_dict = {}
        for p, dhp, t in zip(self.forward_params, dh_params, thetas):
            for i in range(len(p)):
                if i == 0:
                    theta_dict[p[i]] = t
                elif i == 3:
                    a_dict[p[i]] = dhp[i]
                else:
                    d_alpha_dict[p[i]] = dhp[i]
        for vp, v in zip(self.vel_params, vels):
            vel_dict[vp] = v
        for ap, a in zip(self.acc_params, accs):
            acc_dict[ap] = a

        taus = []
        for tau_expr in self.tau_exprs:
            tau = tau_expr.subs(d_alpha_dict).subs(acc_dict).subs(vel_dict).subs(theta_dict).subs(a_dict)
            taus.append(tau)
        return self.tau_exprs, taus

    def _get_vals(self, list_):
        n = len(self.dh_params)
        thetas, lengths, vels, accs = [], [], [], []
        j = 0
        for i in range(len(list_)):
            val = list_[i]
            if 0 <= i < n:
                if self.type[j] == 'r':
                    thetas.append(val) # this value will be optimized
                    lengths.append(self.dh_params[j][3]) # whereas this value won't be optimized as this joint can't change it's length
                elif self.type[j] == 'l':
                    lengths.append(val) # this value will be optimized
                    thetas.append(self.dh_params[j][0]) # whereas this value won't be optimized as this joint can't change its angle
                j += 1
            elif n <= i < 2*n:
                vels.append(val)
            else:
                accs.append(val)
        return thetas, lengths, vels, accs

    def _cost_fn_fd(self, initial_guess, masses, dimensions, taus_req):
        thetas, lengths, vels, accs = self._get_vals(initial_guess)
        taus = self.solve_id(thetas, vels, accs, masses, dimensions, lengths)[1]
        cost = 0
        for tr, t in zip(taus_req, taus):
            cost += (tr - t) ** 2
        cost = cost ** 0.5
        return cost

    # forward dynamics
    def solve_fd(self, masses, dimensions, taus_req, tol=None):
        initial_guess = [0 for _ in range(3*len(self.dh_params))]
        cost_fn = partial(self._cost_fn_fd, masses=masses, dimensions=dimensions, taus_req=taus_req)
        if tol == None:
            tol = 3e-4

        result = minimize(cost_fn, initial_guess, tol=tol)
        optim_vals = result.x
        # optim_vals = initial_guess
        thetas, lengths, vels, accs = self._get_vals(optim_vals)
        taus = self.solve_id(thetas, vels, accs, masses, dimensions, lengths)[1]
        optim_vals = {'thetas': thetas, 'lengths': lengths, 'vels': vels, 'accs': accs}
        return optim_vals, taus
        
    def __repr__(self):
        return f'Robot(name={self.name}, type={self.type})'
