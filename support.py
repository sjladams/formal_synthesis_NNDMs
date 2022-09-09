import numpy as np
import cProfile, pstats, io
import itertools
import math
import os
import torch
from scipy import sparse
import matplotlib.pyplot as plt
from copy import copy

from parameters import STATE_SPACE, DIMS, NEURONS, HIDDEN_LAYERS, ACT_FUNC_TYPE
from nn import train_model, load_model
from systems import ImportSystem
from safety_specifications import SAFETY_SPECS

DIRPATH = os.path.dirname(__file__)


def generate_discretization(dx: np.array, add_full: bool = True):
    axis_range = [np.arange(state_space_elem[0], state_space_elem[1] + dx_elem * 0.5, dx_elem)
                    for state_space_elem, dx_elem in zip(STATE_SPACE, dx)]
    intervals = [list(np.vstack((elem[0:-1], elem[1:])).T) for elem in axis_range]
    rectangles = np.array(list(itertools.product(*intervals)))

    if add_full:
        rectangles = np.concatenate((rectangles, STATE_SPACE[np.newaxis]), axis=0)

    x_centers = np.round(np.average(rectangles, axis=2), 4)
    return rectangles, x_centers


def overlap_discretization(rectangles):
    """
    Sort overlapping rectangles dimension-wise.
    """
    sorted_rectangles = {'intervalsX': dict()}

    for nrDim in range(1, DIMS + 1):
        dim_combis = list(itertools.combinations(range(0, DIMS), r=nrDim))
        sorted_rectangles[nrDim] = dict()
        for dim_combi in dim_combis:
            if len(dim_combi) == 1:
                # \TODO unloop
                dummy = dict()
                for tag, cube in enumerate(rectangles):
                    interval_tag = tuple(tuple(cube[dim]) for dim in dim_combi)
                    if interval_tag in dummy:
                        dummy[interval_tag] += [tag]
                    else:
                        dummy[interval_tag] = [tag]

                sorted_rectangles['intervalsX'][dim_combi[0]] = np.squeeze(np.array(list(dummy.keys())), axis=1)

                sorted_rectangles[nrDim][dim_combi[0]] = np.zeros((len(dummy), rectangles.shape[0]), dtype=bool)
                for countInterval, interval in enumerate(dummy):
                    mask = dummy[interval]
                    sorted_rectangles[nrDim][dim_combi[0]][countInterval, mask] = np.ones(len(mask), dtype=bool)
                sorted_rectangles[nrDim][dim_combi[0]] = sparse.coo_matrix(sorted_rectangles[nrDim][dim_combi[0]])

            else:
                sorted_rectangles[nrDim][dim_combi] = dict()
                for tag, cube in enumerate(rectangles):
                    interval_tag = tuple(tuple(cube[dim]) for dim in dim_combi)
                    if interval_tag in sorted_rectangles[nrDim][dim_combi]:
                        sorted_rectangles[nrDim][dim_combi][interval_tag] += [tag]
                    else:
                        sorted_rectangles[nrDim][dim_combi][interval_tag] = [tag]
    return sorted_rectangles


def group_grid_wrt_rectangles(H_rects, grid_sorting):
    """
    groupding of the discretization w.r.t. to the rectangular-overapproximations of the one-step reachable set of each
    'state', i.e., region in the discretization.
    """

    n = H_rects.shape[1]
    interval_means = [np.mean(grid_sorting['intervalsX'][dim], axis=1) for dim in grid_sorting['intervalsX']]
    groups = list(itertools.product([0, 1, 2], repeat=n))

    sorted_per_dim = {dim: dict() for dim in range(0, n)}
    for dim in range(0, n):
        interval2check = np.tile(interval_means[dim], (H_rects.shape[0], 1))
        sorted_per_dim[dim][0] = sparse.coo_matrix(H_rects[:, dim, 0][:, np.newaxis] >= interval2check)
        sorted_per_dim[dim][1] = sparse.coo_matrix(H_rects[:, dim, 1][:, np.newaxis] <= interval2check)
        sorted_per_dim[dim][2] = sparse.coo_matrix(np.logical_and(H_rects[:, dim, 0][:, np.newaxis] < interval2check,
                                                                   H_rects[:, dim, 1][:,
                                                                   np.newaxis] > interval2check))

    ref_group = tuple(2 for _ in range(n))
    groups_mak = {group: None for group in groups}
    for group in groups:
        dimMasks = []

        for dim in range(0, n):
            dimMasks += [sparse.csr_matrix(sorted_per_dim[dim][group[dim]].dot(grid_sorting[1][dim]))]

        dimMask = dimMasks[0].multiply(dimMasks[1])
        for dim in range(2, n):
            dimMask = dimMask.multiply(dimMasks[dim])

        groups_mak[group] = sparse.coo_matrix(dimMask)

    return groups_mak


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def order_pol_points(pp):
    pp = list(pp)
    cent = (sum([p[0] for p in pp]) / len(pp), sum([p[1] for p in pp]) / len(pp))
    # sort by polar angle
    pp.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))
    return np.array(pp)


def get_crown_bounds(models: dict, hypercubes: np.array, centers: np.array, crown_bounds: dict = None):
    if crown_bounds is None:
        crown_bounds = {tag: dict() for tag in models}

    for tag in models:
        lower_intervals = torch.from_numpy(hypercubes[:, :, 0]).float()
        upper_intervals = torch.from_numpy(hypercubes[:, :, 1]).float()
        (As_L, bs_L), (As_U, bs_U) = models[tag]['nn'].crown_linear(lower_intervals, upper_intervals)

        As_L = As_L.detach().numpy()
        bs_L = bs_L.detach().numpy()
        As_U = As_U.detach().numpy()
        bs_U = bs_U.detach().numpy()

        crown_bounds[tag] = {**crown_bounds[tag], **{tuple(xCenter): {'A_U': As_U[count],
                                                                      'A_L': As_L[count],
                                                                      'b_U': bs_U[count],
                                                                      'b_L': bs_L[count]} for count, xCenter in
                                                     enumerate(centers)}}
    return crown_bounds


def import_model(system_type: str, plot: bool):
    system = ImportSystem(system_type)

    model_name = 'torch_nn_model_{}_dimX{}_ss{}_{}_NEUR{}_LAY{}_ACTF_{}'.format(system_type, DIMS,
                    np.round(STATE_SPACE[:, 0], 2), np.round(STATE_SPACE[:, 1], 2), NEURONS, HIDDEN_LAYERS,
                                                                                ACT_FUNC_TYPE)
    model_path = DIRPATH + f"/models/{model_name}"
    check = os.path.exists(model_path)

    if check:
        model = load_model(model_path, system, plot)
        print(f'used pre-trained model for system type: {system_type}')
    else:
        model = train_model(model_path, system, plot)
    return {'sys': system, 'nn': model}


def generate_domain_limits(ax, plot_x_dims=[0, 1], xtick_freq=None, ytick_freq=None):
    ax.set_xlim(STATE_SPACE[plot_x_dims[0], 0], STATE_SPACE[plot_x_dims[0], 1])
    ax.set_ylim(STATE_SPACE[plot_x_dims[1], 0], STATE_SPACE[plot_x_dims[1], 1])
    if xtick_freq is not None:
        if xtick_freq > STATE_SPACE[plot_x_dims[0], 1] - STATE_SPACE[plot_x_dims[0], 0]:
            ax.set_xticks([])
        else:
            xticks = np.arange(STATE_SPACE[plot_x_dims[0], 0], STATE_SPACE[plot_x_dims[0], 1] + 0.5 * xtick_freq,
                               xtick_freq)
            ax.set_xticks(xticks)
    if ytick_freq is not None:
        if ytick_freq > STATE_SPACE[plot_x_dims[1], 1] - STATE_SPACE[plot_x_dims[1], 0]:
            ax.set_yticks([])
        else:
            yticks = np.arange(STATE_SPACE[plot_x_dims[1], 0], STATE_SPACE[plot_x_dims[1], 1] + 0.5 * ytick_freq,
                               ytick_freq)
            ax.set_yticks(yticks)
    return ax


def cartesian_product(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian_product(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def merge_q_yes_q_no_regions(rectangles: np.array, synthesis: 'Synthesis'):
    # \TODO check assumption that is we don't use LTL, so considering G(..) and F(..) problems, discretizing Qyes doesn't  impact synthesis results..
    if synthesis.use_ltlf:
        Q2check = synthesis.q_no.astype(int)
    else:
        Q2check = np.union1d(synthesis.q_no, synthesis.q_yes).astype(int)

    # check if points are 'fully' connected, i.e. over
    remove2replace = dict()
    for specRegion in SAFETY_SPECS[synthesis.spec_type]:
        for regionTag in SAFETY_SPECS[synthesis.spec_type][specRegion]:
            region = copy(STATE_SPACE)
            for dim in SAFETY_SPECS[synthesis.spec_type][specRegion][regionTag]:
                region[dim] = SAFETY_SPECS[synthesis.spec_type][specRegion][regionTag][dim]

            to_remove = Q2check[np.all(np.logical_and(rectangles[Q2check][:, :, 0] >= region[:, 0],
                                                     rectangles[Q2check][:, :, 1] <= region[:, 1]), axis=1)]

            if to_remove.size:
                remove2replace[tuple([tuple(elem) for elem in region])] = to_remove

    tags2remove = np.concatenate(list(remove2replace.values()))
    new_rectangles = np.array(list(remove2replace.keys()))
    new_centers = np.round(np.average(new_rectangles, axis=2), 4)
    return tags2remove, new_rectangles, new_centers


def DEBUG_PLOT(grids, checker):
    models = list(grids.grids.keys())
    modelTag = models[0]

    test = grids.grids[modelTag].transBoundsTerm2
    regionTag = 20
    origin = grids.rectangles[regionTag]
    overapprox = grids.grids[modelTag].DEBUG_overapprox[regionTag]
    upperbox = grids.grids[modelTag].DEBUG_boundU[regionTag]
    lowerbox = grids.grids[modelTag].DEBUG_boundL[regionTag]

    plot_dims = [0, 2]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlim(0, 2)

    # ax = generate_domain_limits(ax, plot_x_dims=plot_dims)
    patch = plt.Polygon(order_pol_points(cartesian_product(origin[plot_dims])),
                        True, color='red', alpha=0.8, fill=False)
    ax.add_patch(patch)
    patch = plt.Polygon(order_pol_points(cartesian_product(overapprox[plot_dims])),
                        True, color='green', alpha=0.8, fill=False)
    ax.add_patch(patch)
    patch = plt.Polygon(order_pol_points(upperbox[:, plot_dims]),
                        True, color='black', alpha=0.8, fill=False)
    ax.add_patch(patch)
    patch = plt.Polygon(order_pol_points(lowerbox[:, plot_dims]),
                        True, color='black', alpha=0.8, fill=False)
    ax.add_patch(patch)
    # ax.axis('equal')
    plt.show()

###


# def TEST_CROWN_PERF(imdp):
#     models = list(imdp.imcs.keys())
#     modelTag = models[0]
#
#     # Compute total covering surfaces of all regions
#     total = 0
#     for tag in range(0,imdp.rectangles.shape[0]-1):
#         overapprox = imdp.imcs[modelTag].TEST_CROWN_PERF['H_rects'][tag]
#         total += np.product(overapprox[:,1]-overapprox[:,0])
#     print('NEUR: {}, LAYER: {}, SYSTEM: {}: total covered surface: {}'.format(NEURONS, HIDDEN_LAYERS, modelTag, total))
#
#     regionTag = 50
#     origin = imdp.rectangles[regionTag]
#
#     overapprox = imdp.imcs[modelTag].TEST_CROWN_PERF['H_rects'][regionTag]
#     upperbox = imdp.imcs[modelTag].TEST_CROWN_PERF['H_l'][regionTag]
#     lowerbox = imdp.imcs[modelTag].TEST_CROWN_PERF['H_u'][regionTag]
#
#     plot_dims = [0, 1]
#     fig, ax = plt.subplots(figsize=(4, 4))
#     ax.set_ylim(-1.75, -1)
#     ax.set_xlim(-1.3, -0.5)
#
#     # ax = generate_domain_limits(ax, plot_x_dims=plot_dims)
#     patch = plt.Polygon(order_pol_points(cartesian_product(origin[plot_dims])),
#                         True, color='red', alpha=0.8, fill=False)
#     ax.add_patch(patch)
#     patch = plt.Polygon(order_pol_points(cartesian_product(overapprox[plot_dims])),
#                         True, color='green', alpha=0.8, fill=False)
#     ax.add_patch(patch)
#     patch = plt.Polygon(order_pol_points(upperbox[:, plot_dims]),
#                         True, color='black', alpha=0.8, fill=False)
#     ax.add_patch(patch)
#     patch = plt.Polygon(order_pol_points(lowerbox[:, plot_dims]),
#                         True, color='black', alpha=0.8, fill=False)
#     ax.add_patch(patch)
#     plt.show()
#
#     x=1