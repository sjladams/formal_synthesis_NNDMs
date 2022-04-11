import os
import torch
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from copy import copy
from numba import jit
import subprocess
import matplotlib.colors as mcolors

from support import order_pol_points, generate_domain_limits, cartesian_product
from safety_specifications import SAFETY_SPECS
from parameters import STATE_SPACE

DIRPATH = os.path.dirname(__file__)

class Synthesis:
    def __init__(self, imdp: 'IMDP', spec_type: str, k: int = 10, p: float = 0.95, bound: str = '>=',
                 use_cpp: bool = False, proj_dims: list = [0, 1], use_LTL: bool = False,
                 dest_tag: str = 'D', obs_tag: str = 'O'):
        self.use_ltlf = use_LTL
        self.proj_dims = proj_dims
        self.spec_type = spec_type
        self.k = k
        self.p = p
        self.bound = bound
        self.use_cpp = use_cpp

        self.imdp = imdp
        self.dest_tag = dest_tag
        self.obs_tag = obs_tag

        self.name = 'example'

        # \TODO add automatic LTL formula generation
        if self.use_ltlf:
            # -- LTL ------------------------------------------------------------------------------------------------------
            # self.P_masks_LTL = {'D': np.array([[0, 1, 0],
            #                                             [0, 1, 0],
            #                                             [0, 0, 1]]),
            #                   'O': np.array([[0, 0, 1],
            #                                          [0, 0, 1],
            #                                          [0, 0, 1]]),
            #                   'none': np.array([[1, 0, 0],
            #                                     [0, 1, 0],
            #                                     [0, 0, 1]])}
            self.P_masks_LTL = {'D1': np.array([[0, 0, 1, 0, 0],
                                                [0, 0, 0, 1, 0],
                                                [0, 0, 1, 0, 0],
                                                [0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 1]]),
                                'D2': np.array([[0, 1, 0, 0, 0],
                                                [0, 1, 0, 0, 0],
                                                [0, 0, 0, 1, 0],
                                                [0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 1]]),
                                'O': np.array([[0, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 1]]),
                                'none': np.array([[1, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0],
                                                  [0, 0, 1, 0, 0],
                                                  [0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 1]])}
            self.basis_Vmin = np.array([0, 0, 0, 1, 0])
            self.nr_S_states = list(self.P_masks_LTL.values())[0].shape[0]

        # -----------------------------------------------------------------------------------------------------------
        self.update_grid_elements()
        self.init_checking_state_regions()
        self.update_grid_elements()

    def update_grid_elements(self):
        self.nr_actions = len(self.imdp.imcs)
        self.nr_states = self.imdp.rectangles.shape[0]
        self.init_steps()

        if hasattr(self, 'checking_states'):
            self.init_labels()
            self.find_q_yes_q_no() # \TODO Disable when we're not running ltlf
            if self.use_ltlf:
                self.init_ltlf_steps()

    def init_checking_state_regions(self):
        region_types = list(SAFETY_SPECS[self.spec_type].keys())
        self.checking_states = dict()
        for regionType in region_types:
            self.checking_states[regionType] = dict()
            for tag in SAFETY_SPECS[self.spec_type][regionType]:
                self.checking_states[regionType][tag] = copy(STATE_SPACE)
                for dim in SAFETY_SPECS[self.spec_type][regionType][tag]:
                    self.checking_states[regionType][tag][dim] = SAFETY_SPECS[self.spec_type][regionType][tag][dim]

    def init_labels(self):
        self.labels = {state: np.zeros(self.imdp.rectangles.shape[0]) for state in self.checking_states}
        for tag, cube in enumerate(self.imdp.rectangles[:-1]):
            for regionType in self.checking_states:
                for regionTag in self.checking_states[regionType]:
                    if check_overlap_Intervals(cube, self.checking_states[regionType][regionTag]):
                        self.labels[regionType][tag] = 1

        label_values = np.array(list(self.labels.values()))
        self.labels['none'] = (np.sum(label_values, axis=0) == 0).astype(int)

    def find_q_yes_q_no(self):
        q_yes = np.zeros(self.nr_states, dtype=bool)
        q_no = np.zeros(self.nr_states, dtype=bool)

        for region_type in SAFETY_SPECS[self.spec_type]:
            if self.dest_tag in region_type:
                q_yes = np.logical_or(q_yes, self.labels[region_type])
            elif self.obs_tag in region_type:
                q_no = np.logical_or(q_no, self.labels[region_type])
        self.q_yes = np.where(q_yes)[0]
        self.q_no = np.where(q_no)[0]

    def init_steps(self):
        self.steps_min = np.zeros((self.nr_actions * self.nr_states, self.nr_states))
        self.steps_max = np.zeros((self.nr_actions * self.nr_states, self.nr_states))

        for action, tag in enumerate(self.imdp.imcs):
            self.steps_min[np.arange(action, self.nr_states * self.nr_actions, self.nr_actions), :] = \
                self.imdp.imcs[tag].trans_bounds[:, :, 0]
            self.steps_max[np.arange(action, self.nr_states * self.nr_actions, self.nr_actions), :] = \
                self.imdp.imcs[tag].trans_bounds[:, :, 1]

    def init_ltlf_steps(self):
        to_fill = np.zeros((self.nr_S_states * self.nr_states, self.nr_S_states))
        for tag in self.P_masks_LTL:
            to_fill += np.kron(self.labels[tag][np.newaxis].T, self.P_masks_LTL[tag])

        self.steps_min_ltlf = np.zeros(
            (self.nr_actions * self.nr_states * self.nr_S_states, self.nr_states * self.nr_S_states))
        self.steps_max_ltlf = np.zeros(
            (self.nr_actions * self.nr_states * self.nr_S_states, self.nr_states * self.nr_S_states))

        for state in range(0, self.nr_actions):
            P_min_state = self.steps_min[np.arange(state, self.nr_states * self.nr_actions, self.nr_actions), :]
            P_max_state = self.steps_max[np.arange(state, self.nr_states * self.nr_actions, self.nr_actions), :]

            P_min_prod = np.multiply(np.tile(to_fill, self.nr_states),
                                     np.kron(P_min_state, np.ones((self.nr_S_states, self.nr_S_states))))
            P_max_prod = np.multiply(np.tile(to_fill, self.nr_states),
                                     np.kron(P_max_state, np.ones((self.nr_S_states, self.nr_S_states))))

            self.steps_min_ltlf[np.arange(state, self.nr_S_states * self.nr_states * self.nr_actions,
                                          self.nr_actions), :] = P_min_prod
            self.steps_max_ltlf[np.arange(state, self.nr_S_states * self.nr_states * self.nr_actions,
                                          self.nr_actions), :] = P_max_prod

    def run_synthesis(self):
        if self.use_ltlf:
            self.solve_for_ltlf()
        else:
            self.bounded_until()
        self.apply_projection()

    def solve_for_ltlf(self):
        v_min_init = np.tile(self.basis_Vmin, self.nr_states)
        v_min = copy(v_min_init)
        v_max = copy(v_min_init)

        policy = np.zeros(v_min.T.shape)
        policy_hist = np.zeros((v_min.shape[0], self.k))

        for tk in list(range(0, self.k))[::-1]:
            # Get the best and worst MDPs
            ind_ascend = v_min.argsort()
            ind_descend = v_min.argsort()[::-1]

            # \TODO numpy -> torch full synthesis code
            min_mdp, max_mdp = value_iteration(torch.from_numpy(self.steps_min_ltlf),
                                               torch.from_numpy(self.steps_max_ltlf),
                                               ind_ascend, ind_descend)

            ind_min = np.matmul(min_mdp.detach().numpy(), v_min)
            ind_max = np.matmul(max_mdp.detach().numpy(), v_max)

            ctrl = np.zeros(self.steps_max_ltlf.shape[1], dtype=int)
            if self.nr_actions > 1:
                # maximize the lower bound and find the policy
                ind_min = np.reshape(ind_min, (self.nr_actions, int(len(ind_min) / self.nr_actions)), order='F')
                ind_max = np.reshape(ind_max, (self.nr_actions, int(len(ind_max) / self.nr_actions)), order='F')

                v_min = np.max(ind_min, axis=0)

                control_by_min = np.sum(ind_min, axis=0) != 0
                ctrl[control_by_min] = np.argmax(ind_min[:, control_by_min], axis=0)
                ctrl[~control_by_min] = np.argmax(ind_max[:, ~control_by_min], axis=0)
            else:
                v_min = ind_min
                ctrl = policy.astype(int)

            policy_hist[:, tk] = ctrl
            P_max = max_mdp[np.array(range(0, len(ctrl))) * self.nr_actions + ctrl, :]
            v_max = np.matmul(P_max, v_max)

            v_min = np.clip(v_min + v_min_init, 0, 1)
            v_max = np.clip(v_max + v_min_init, 0, 1)

        # Select only s1 states (start there)
        self.v_min = v_min[::self.nr_S_states]
        self.v_max = v_max[::self.nr_S_states]

        ind_min, ind_max = self.classify_states(self.v_min, self.v_max)

        self.v_min_clas = np.array([ind_min, self.v_min[ind_min]]).T
        self.v_max_clas = np.array([ind_max, self.v_max[ind_max]]).T

        self.q_yes = np.array([])
        self.q_sat = np.intersect1d(ind_min, ind_max)
        self.q_possible = np.setdiff1d(np.union1d(ind_min, ind_max), self.q_sat)
        self.q_no = np.setdiff1d(np.arange(0, self.nr_states), np.union1d(self.q_sat, self.q_possible))
        self.q_no = self.q_no[self.q_no != self.nr_states - 1]

        self.policy = policy_hist[::self.nr_S_states].astype(int)

    def classify_states(self, v_min: np.array, v_max: np.array, by_torch: bool = False):
        if self.bound == '>':
            if by_torch:
                return torch.where(v_min > self.p)[0], torch.where(v_max > self.p)[0]
            else:
                return np.where(v_min > self.p)[0], np.where(v_max > self.p)[0]
        elif self.bound == '>=':
            if by_torch:
                return torch.where(v_min >= self.p)[0], torch.where(v_max >= self.p)[0]
            else:
                return np.where(v_min >= self.p)[0], np.where(v_max >= self.p)[0]
        elif self.bound == '<':
            if by_torch:
                return torch.where(v_min < self.p)[0], torch.where(v_max < self.p)[0]
            else:
                return np.where(v_min < self.p)[0], np.where(v_max < self.p)[0]
        elif self.bound == '<=':
            if by_torch:
                return torch.where(v_min <= self.p)[0], torch.where(v_max <= self.p)[0]
            else:
                return np.where(v_min <= self.p)[0], np.where(v_max <= self.p)[0]

    def bounded_until(self):
        steps_max = torch.from_numpy(copy(self.steps_max))
        steps_min = torch.from_numpy(copy(self.steps_min))

        if self.use_cpp:
            filename = '{}.txt'.format(self.name)
            with open(filename, 'w') as f:
                f.write('{} \n'.format(self.nr_states))
                f.write('{} \n'.format(self.nr_actions))
                f.write('{} \n'.format(len(self.q_yes)))
                for i in range(0, len(self.q_yes)):
                    f.write('{} '.format(self.q_yes[i]))
                f.write('\n')
                for i in range(0, self.nr_states):
                    if i not in self.q_no:
                        for a in range(0, self.nr_actions):
                            ij = np.where(steps_max[i * self.nr_actions + a, :])[0]
                            if np.sum(steps_max[i * self.nr_actions + a, :]) < 1:
                                remain = 1 - np.sum(steps_max[i * self.nr_actions + a, :])
                                steps_max[i * self.nr_actions + a, ij[-1]] = steps_max[
                                                                                 i * self.nr_actions + a, ij[
                                                                                     -1]] + remain
                            for j in ij:
                                f.write('{} {} {} {:.4f} {:.4f}'.format(i, a, j,
                                                                        steps_min[i * self.nr_actions + a, j],
                                                                        steps_max[i * self.nr_actions + a, j]))
                                if (i < self.nr_states or j < ij[-1] or a < self.nr_actions):
                                    f.write(' \n')
                    else:
                        f.write('{} {} {} {} {}'.format(i, 0, i, 1., 1.))
                        if i < self.nr_states:
                            f.write(' \n')

            if self.bound == '>=' or self.bound == '>' or self.bound == 'max':
                minmax = 'maximize pessimistic'
            else:
                minmax = 'minimize pessimistic'

            s = "{}/synthesis.exe {} {} 0.001 {}/{}".format(DIRPATH, minmax, self.k, DIRPATH, filename)
            process = subprocess.Popen(s, stdout=subprocess.PIPE, shell=True)
            output, error = process.communicate()

            output = output.decode("utf-8")
            output = output.split('\r\n')
            output = [elem for elem in output if not 'Warning' in elem]
            output = [elem.split(' ') for elem in output if (elem != '')]
            output = np.array(output).astype(float)

            policy = output[:, 1] + 1
            policy[policy == 0] = 1
            v_min = output[:, 2]
            v_max = output[:, 3]
        else:
            v_min = torch.zeros(steps_min.shape[1])
            v_max = torch.zeros(steps_max.shape[1])

            v_min[self.q_yes] = 1
            v_max[self.q_yes] = 1

            policy = torch.zeros(v_min.T.shape)
            policy_hist = torch.zeros((v_min.shape[0], self.k))

            if self.bound == '>=' or self.bound == '>' or self.bound == 'max':
                for tk in list(range(0, self.k))[::-1]:
                    # The best and worst MDPs
                    ind_ascend = v_min.argsort()
                    ind_descend = ind_ascend.flip(dims=[0])

                    min_mdp, max_mdp = value_iteration(steps_min, steps_max, ind_ascend, ind_descend)
                    ind_min = torch.matmul(min_mdp, v_min)
                    ind_max = torch.matmul(max_mdp, v_max)

                    ctrl = torch.zeros(steps_max.shape[1], dtype=int)
                    if self.nr_actions > 1:
                        # maximize the lower bound and find the policy
                        ind_min = reshape_fortran(ind_min, (self.nr_actions, int(len(ind_min) / self.nr_actions)))
                        ind_max = reshape_fortran(ind_max, (self.nr_actions, int(len(ind_max) / self.nr_actions)))

                        v_min = torch.max(ind_min, axis=0)[0]

                        control_by_min = torch.sum(ind_min, axis=0) != 0
                        ctrl[control_by_min] = torch.argmax(ind_min[:, control_by_min], axis=0)
                        ctrl[~control_by_min] = torch.argmax(ind_max[:, ~control_by_min], axis=0)
                    else:
                        v_min = ind_min
                        ctrl = policy.type(torch.int64)

                    policy_hist[:, tk] = ctrl
                    P_max = max_mdp[torch.tensor(range(0, len(ctrl))) * self.nr_actions + ctrl, :]

                    v_max = torch.matmul(P_max, v_max)

                    # update the probs of q_yes and q_no
                    v_min[self.q_yes] = 1
                    v_min[self.q_no] = 0

                    v_max[self.q_yes] = 1
                    v_max[self.q_no] = 0

            policy = policy_hist

        ind_min, ind_max = self.classify_states(v_min, v_max, by_torch=True)

        self.v_min_clas = np.array([ind_min.numpy(), v_min[ind_min].numpy()]).T
        self.v_max_class = np.array([ind_max.numpy(), v_max[ind_max].numpy()]).T

        self.q_sat = np.union1d(np.intersect1d(ind_min.numpy(), ind_max.numpy()), self.q_yes)  # \TODO to function
        self.q_possible = np.setdiff1d(np.union1d(ind_min.numpy(), ind_max.numpy()), np.union1d(self.q_sat, self.q_no))
        self.q_no = np.setdiff1d(np.arange(0, self.nr_states), np.union1d(self.q_sat, self.q_possible))
        self.q_no = self.q_no[self.q_no != self.nr_states - 1]

        self.v_min = v_min.numpy()
        self.v_max = v_max.numpy()
        self.policy = policy.numpy().astype(int)

    def apply_projection(self):
        if self.imdp.rectangles.shape[1] <= 2:
            self.projection = {tuple(tuple(elem) for elem in rect): tag for tag, rect in
                               enumerate(self.imdp.rectangles[:-1])}  # \TODO if pojection needed, instead if-state and use rect?
            self.projection_rects = np.array(list(self.projection.keys()))
            self.v_min_proj = self.v_min
            self.v_max_proj = self.v_max
            self.q_sat_proj = self.q_sat
            self.q_possible_proj = self.q_possible
            self.q_no_proj = self.q_no
            self.q_yes_proj = self.q_yes
        else:  # \TODO Vectorize
            projection = dict()
            for region in self.imdp.grid_sorting[2][tuple(self.proj_dims)]:
                if region not in projection:
                    included_rects = list(projection.keys())
                    if len(included_rects) != 0:
                        mask_incl = np.logical_and(np.array(region)[:, 0] <= np.array(included_rects)[:, :, 0],
                                                   np.array(region)[:, 1] >= np.array(included_rects)[:, :, 1])
                        mask_incl = np.all(mask_incl, axis=1)
                        mask_cap = np.logical_and(np.array(region)[:, 0] >= np.array(included_rects)[:, :, 0],
                                                  np.array(region)[:, 1] <= np.array(included_rects)[:, :, 1])
                        mask_cap = np.where(np.all(mask_cap, axis=1))[0]

                        # check:
                        if mask_cap.size > 1:
                            raise ValueError("can by construction not been captured by more than one region")
                        if np.all(mask_incl):
                            pass  # this is the outer region, we don't include it in the projected states

                        elif np.any(mask_incl):
                            to_remove = [included_rects[elem] for elem in np.where(mask_incl)[0]]
                            projection[region] = self.imdp.grid_sorting[2][tuple(self.proj_dims)][region]
                            for elem in to_remove:
                                projection[region] += projection[elem]
                                del projection[elem]
                        elif mask_cap.size == 1:
                            projection[included_rects[mask_cap[0]]] += self.imdp.grid_sorting[2][
                                tuple(self.proj_dims)][region]
                        else:
                            projection[region] = self.imdp.grid_sorting[2][tuple(self.proj_dims)][region]
                    else:
                        projection[region] = self.imdp.grid_sorting[2][tuple(self.proj_dims)][region]

            self.projection = projection
            self.projection_rects = np.array(list(projection.keys()))
            self.v_min_proj = np.clip(np.array([np.sum(self.v_min[projection[region]]) for region in projection]), 0, 1)
            self.v_max_proj = np.clip(np.array([np.sum(self.v_max[projection[region]]) for region in projection]), 0, 1)

            ind_min, ind_max = self.classify_states(self.v_min_proj, self.v_max_proj)

            self.q_sat_proj = np.intersect1d(ind_min, ind_max)
            self.q_possible_proj = np.setdiff1d(np.union1d(ind_min, ind_max), self.q_sat_proj)
            self.q_no_proj = np.setdiff1d(np.arange(0, self.projection_rects.shape[0]),
                                          np.union1d(self.q_possible_proj, self.q_sat_proj))

            self.q_yes_proj = np.array([])
            for q_yes in self.q_yes:
                region = self.imdp.rectangles[q_yes][self.proj_dims, :]
                self.q_yes_proj = np.append(self.q_yes_proj, np.where(np.all(np.all(
                    self.projection_rects == region, axis=2), axis=1))[0])

    def plot(self, models: dict, mark_des: bool = False, mark_obs: bool = False, plot_sims: bool = False,
             nr_sims: int = 0, sim_states_proj: list = [], plot_des_tags: bool = False,
             plot_type: int = 'classification', xtick_freq: int = 1, ytick_freq: int = 1, save: bool = False,
             name: str = 'test', fig_size: tuple = (4, 4),
             plot_legend: bool = False, legend_size: float = 2.5, legend_pad: float = 3):

        spec_tags = list(SAFETY_SPECS[self.spec_type].keys())
        dest_tags = [tag for tag in spec_tags if self.dest_tag in tag]
        obs_tags = [tag for tag in spec_tags if self.obs_tag in tag]

        fig, ax = plt.subplots(figsize=fig_size)
        ax = generate_domain_limits(ax, plot_x_dims=self.proj_dims, xtick_freq=xtick_freq, ytick_freq=ytick_freq)

        # simulations --------------------------------------------------------------------------------------------------
        text = {'x1 location': [], 'x2 location': [], 'action': []}
        if plot_sims:
            des_rects = [np.array(list(self.checking_states[tag].values())) for tag in dest_tags]
            if not sim_states_proj:
                sim_states_proj = np.setxor1d(self.q_sat_proj, self.q_yes_proj).astype(int)
                if nr_sims == 0 or nr_sims > len(sim_states_proj):
                    nr_sims = len(sim_states_proj)
                sim_states_proj = np.random.choice(sim_states_proj, size=nr_sims, replace=False)
            else:
                sim_states_proj = np.array(sim_states_proj)

            for sim_state_proj in sim_states_proj:
                # find all corresponding states in original space
                sim_states = np.array(self.projection[tuple(tuple(elem) for elem in
                                                            self.projection_rects[sim_state_proj])])
                # sort these based on indVmin
                sim_states = sim_states[self.v_min[sim_states].argsort()[::-1]]
                success = False

                for sim_state in sim_states:
                    visited_states = np.zeros(len(dest_tags), bool)
                    actions = []
                    track = np.mean(self.rectangles[sim_state], axis=1)[np.newaxis]
                    for i in range(0, self.k):
                        current_state = None
                        # which tag we are in:
                        mask = dict()
                        for dim in range(0, n):
                            intervals = self.imdp.gird_sorting['intervalsX'][dim][:-1]
                            mask[dim] = np.where(np.logical_and(
                                intervals[:, 0] <= track[i][dim], intervals[:, 1] >= track[i][dim]))[0]
                        for combi in list(it.product(*list(mask.values()))):
                            region = tuple(tuple(self.imdp.grid_sorting['intervalsX'][dim][elem]) for dim, elem in
                                           enumerate(combi))
                            try:
                                current_state = \
                                    self.imdp.gird_sorting[n][tuple(dim for dim in range(0, n))][tuple(region)][0]
                                break
                            except:
                                continue
                            break
                        if current_state is None:
                            print("can't print path of state {}".format(sim_state))
                            break

                        action = self.policy[current_state][i]
                        actions += [action]
                        action_ref = list(models.keys())[action]
                        next_loc = np.array([models[action_ref]['sys'].generator(*track[i], dim=dim) for
                                              dim in range(0, n)])
                        track = np.append(track, next_loc[np.newaxis], axis=0)
                        for tag, des_rect in enumerate(des_rects):
                            if check_overlap_point_rects(next_loc, des_rect):
                                visited_states[tag] = True

                        if np.all(visited_states):
                            plt.plot(track[:, self.proj_dims[0]], track[:, self.proj_dims[1]], 'b-', linewidth=1)
                            plt.plot(track[0, self.proj_dims[0]], track[0, self.proj_dims[1]], 'b.', markersize=4)
                            text['x1 location'] += list(track[:-1, 0])
                            text['x2 location'] += list(track[:-1, 1])
                            text['action'] += actions
                            success = True
                            break
                    if success:
                        break

        if plot_type == 'classification':
            # Plot Classification --------------------------------------------------------------------------------------
            for tag in self.q_sat_proj:
                vertices = cartesian_product(np.array(self.projection_rects[tag]))
                patch = plt.Polygon(order_pol_points(vertices), True, color='green', alpha=0.8, fill=True)
                ax.add_patch(patch)
            for tag in self.q_possible_proj:
                vertices = cartesian_product(np.array(self.projection_rects[tag]))
                patch = plt.Polygon(order_pol_points(vertices), True, color='orange', alpha=0.8, fill=True)
                ax.add_patch(patch)
            for tag in self.q_no_proj:
                vertices = cartesian_product(np.array(self.projection_rects[tag]))
                patch = plt.Polygon(order_pol_points(vertices), True, color='red', alpha=0.8, fill=True)
                ax.add_patch(patch)
        else:
            # Plot Lower bounds ----------------------------------------------------------------------------------------
            if plot_type == 'lower bounds':
                bounds = self.v_min_proj
            elif plot_type == 'upper bounds':
                bounds = self.v_max_proj

            c = mcolors.ColorConverter().to_rgb
            cmap = make_colormap(
                [c('red'), c('green')])
            # cmap = get_cmap('RdYlGn') # 'gray'
            min_ref = 0.
            max_ref = np.max(bounds)

            for bound, region in zip(bounds, self.projection_rects):
                vertices = cartesian_product(region)
                pgon_S_tilde = plt.Polygon(order_pol_points(vertices), True,
                                           color=cmap(bound - min_ref / (max_ref - min_ref)), alpha=0.8, fill=True)
                ax.add_patch(pgon_S_tilde)

        if mark_des:
            for dest_tag in dest_tags:
                for i in self.checking_states[dest_tag]:
                    patch = plt.Polygon(order_pol_points(cartesian_product(self.checking_states[dest_tag][i])[:,
                                                         self.proj_dims]),
                                        True, color='black', fill=False)
                    ax.add_patch(patch)
                    if plot_des_tags:
                        center = np.mean(self.checking_states[dest_tag][i], axis=1)[self.proj_dims]
                        plt.text(*center, dest_tag, horizontalalignment='center', fontsize=12,
                                 verticalalignment='center')
        if mark_obs:
            for obs_tag in obs_tags:
                for i in self.checking_states[obs_tag]:
                    patch = plt.Polygon(order_pol_points(cartesian_product(self.checking_states[obs_tag][i])[:,
                                                         self.proj_dims]),
                                        True, color='black', fill=False)
                    ax.add_patch(patch)
                    if plot_des_tags:
                        center = np.mean(self.checking_states[obs_tag][i], axis=1)[self.proj_dims]
                        plt.text(*center, obs_tag, horizontalalignment='center', fontsize=12,
                                 verticalalignment='center')

        if plot_type == 'classification' and plot_legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
            legend_elements = [Patch(facecolor='green', label='$Q^{yes}$'),
                               Patch(facecolor='orange', label='$Q^{?}$'),
                               Patch(facecolor='red', label='$Q^{no}$')]
            ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1, 0.8, 0.1), ncol=3, edgecolor='1.0')
        elif plot_legend:
            # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_ref, vmax=max_ref+0.0000001))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0., vmax=1.))
            sm.set_array([])
            the_divider = make_axes_locatable(ax)
            color_axis = the_divider.append_axes("top", size="{}%".format(legend_size),
                                                 pad="{}%".format(legend_pad))
            color_axis.yaxis.set_visible(False)
            plt.colorbar(sm, orientation='horizontal', cax=color_axis, ticks=[0, 0.5, 1.])
            color_axis.xaxis.set_ticks_position('top')

        # plot actions
        for i, _ in enumerate(text['action']):
            ax.text(text['x1 location'][i],
                    text['x2 location'][i],
                    '${}$'.format(text['action'][i]),
                    horizontalalignment='right', fontsize=10, fontweight='bold')

        ax.set_aspect('equal')
        plt.show()


def check_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) > 0


def check_overlap_Intervals(a, b):
    if a.shape[0] >= b.shape[0]:
        for dim in range(0, a.shape[0]):
            check = True
            if not check_overlap(a[dim], b[dim]):
                check = False
                break
    if a.shape[0] < b.shape[0]:
        for dim in range(0, b.shape[0]):
            check = True
            if not check_overlap(a[dim], b[dim]):
                check = False
                break
    return check


def check_overlap_point_rects(point, cubes):
    for cube in cubes:
        if np.all(point >= cube[:, 0]) and np.all(point <= cube[:, 1]):
            return True
    return False


def value_iteration(steps_min: torch.tensor, steps_max: torch.tensor, ind_ascend: torch.tensor,
                    ind_descend: torch.tensor):
    min_mdp = torch.zeros(steps_min.shape)
    max_mdp = torch.zeros(steps_max.shape)

    for i in range(0, steps_min.shape[0]):
        cols = torch.where(steps_max[i])[0]

        if type(ind_ascend).__module__ != np.__name__:  # \TODO ensure
            ind_ascend = ind_ascend.numpy()
        if type(ind_descend).__module__ != np.__name__:
            ind_descend = ind_descend.numpy()

        ind_ascend_sparse = intersect_preserve_order(ind_ascend, cols.numpy())
        ind_descend_sparse = intersect_preserve_order(ind_descend, cols.numpy())

        data_ascend = torch.cat((steps_min[i][ind_ascend_sparse][None], steps_max[i][ind_ascend_sparse][None]), axis=0)
        data_descend = torch.cat((steps_min[i][ind_descend_sparse][None], steps_max[i][ind_descend_sparse][None]),
                                 axis=0)

        output_ascend = get_true_trans_prob_sparse(data_ascend.numpy())
        min_mdp[i, ind_ascend_sparse] = torch.from_numpy(output_ascend).type(torch.float32)
        output_descend = get_true_trans_prob_sparse(data_descend.numpy())
        max_mdp[i, ind_descend_sparse] = torch.from_numpy(output_descend).type(torch.float32)

    return min_mdp, max_mdp


@jit(nopython=True)
def intersect_preserve_order(a, b):
    dummy = np.full((a.shape[0],), False)
    dummy[b] = True
    return a[dummy[a]]


@jit(nopython=True)
def get_true_trans_prob_sparse(data):
    p = np.zeros(data.shape[1])
    used = np.sum(data[0, :])
    remain = 1 - used

    for i in range(0, data.shape[1]):
        if data[1, i] <= (remain + data[0, i]):
            p[i] = data[1, i]
        else:
            p[i] = data[0, i] + remain
        remain = np.max(np.array([0, remain - (data[1, i] - data[0, i])]))
    return p


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))