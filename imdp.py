import numpy as np
import itertools
from copy import copy
import torch

from support import overlap_discretization, cartesian_product, group_grid_wrt_rectangles


class IMDP:
    def __init__(self, rectangles: np.array, centers: np.array, crown_bounds_all: dict, std: np.array):
        self.rectangles = rectangles
        self.centers = centers
        self.grid_sorting = overlap_discretization(rectangles)
        self.vertices = np.array(list(map(cartesian_product, self.rectangles)))
        self.crown_bounds_all = crown_bounds_all
        self.std = std

        # Compute Bounds on Transition Probabilities
        self.imcs = dict()
        for tag in crown_bounds_all:
            imc = IMC(tag)
            imc.A_u, imc.A_l, imc.b_u, imc.b_l = get_explicit_bounds(crown_bounds_all[tag], centers)
            imc.trans_bounds = bound_transition_kernel(imc.A_u, imc.A_l, imc.b_u, imc.b_l, self.vertices,
                                                       self.centers, self.rectangles, self.grid_sorting, self.std)
            imc.trans_bounds[:, -1, :] = np.array([1 - imc.trans_bounds[:, -1, 1], 1 - imc.trans_bounds[:, -1, 0]]).T
            imc.trans_bounds[-1, :, :] = 0
            imc.trans_bounds[-1, -1, :] = 1

            # TEMP
            # imc.TEST_CROWN_PERF = TEST_CROWN_PERF

            self.imcs[tag] = imc

    def merge(self, tags2remove: np.array, new_rectangles: np.array, new_centers: np.array, crown_bounds_all: dict):
        new_vertices = np.array(list(map(cartesian_product, new_rectangles)))
        new_grid_sorting = overlap_discretization(new_rectangles)

        for tag in self.imcs:
            self.adapt_grid(self.imcs[tag], tags2remove, crown_bounds_all[tag], new_centers, new_vertices,
                            new_grid_sorting, new_rectangles)

        self.rectangles = replace_elems(self.rectangles, new_rectangles, tags2remove)
        self.centers = replace_elems(self.centers, new_centers, tags2remove)
        self.vertices = replace_elems(self.vertices, new_vertices, tags2remove)
        self.grid_sorting = overlap_discretization(self.rectangles)

    def adapt_grid(self, imc, tags2remove, crown_bounds, new_centers, new_vertices, new_grid_sorting, new_rectangles):
        trans_bounds_2be_refined = copy(imc.trans_bounds)

        ## -- Run CROWN for hypercubesXRefined -------------------------------------------------------------------------
        A_u, A_l, b_u, b_l = get_explicit_bounds(crown_bounds, new_centers)

        ## -- term 2 ---------------------------------------------------------------------------------------------------
        # (OLD) transBoundsTerms \in (|S| x |STile|)
        # part1 \in (|SRefined| x |STilde|)
        # part2 \in (|S| x |STildeRefined|)
        # part3 \in (|SRefined| x |STildeRefined|)

        trans_bounds_part1 = bound_transition_kernel(A_u, A_l, b_u, b_l, new_vertices, self.centers, self.rectangles,
                                                     self.grid_sorting, self.std)
        trans_bounds_part1[:, -1, :] = np.array([1 - trans_bounds_part1[:, -1, 1], 1 - trans_bounds_part1[:, -1, 0]]).T

        trans_bounds_part2 = bound_transition_kernel(imc.A_u, imc.A_l, imc.b_u, imc.b_l, self.vertices, new_centers,
                                                     new_rectangles, new_grid_sorting, self.std)
        trans_bounds_part2[-1, :, :] = 0

        trans_bounds_part3 = bound_transition_kernel(A_u, A_l, b_u, b_l, new_vertices, new_centers, new_rectangles,
                                                     new_grid_sorting, self.std)

        # First merge, then remove, because part 2 also contains the old x regions, is possible because we append (-1)
        # the refined values
        old_merged_part2 = np.concatenate((trans_bounds_2be_refined[:, :-1], trans_bounds_part2,
                                           trans_bounds_2be_refined[:, -1][:, np.newaxis]), axis=1)
        part1_merged_part3 = np.concatenate((trans_bounds_part1[:, :-1], trans_bounds_part3,
                                             trans_bounds_part1[:, -1][:, np.newaxis]), axis=1)
        trans_bounds_2be_refined = np.concatenate((old_merged_part2[:-1], part1_merged_part3,
                                                   old_merged_part2[-1][np.newaxis]), axis=0)
        trans_bounds_2be_refined = np.delete(trans_bounds_2be_refined, tags2remove, axis=0)
        trans_bounds_2be_refined = np.delete(trans_bounds_2be_refined, tags2remove, axis=1)

        ## -- Adapt A_U, A_L, b_U, b_l --------------------------------------------------------------------------------
        imc.A_u = replace_elems(imc.A_u, A_u, tags2remove)
        imc.A_l = replace_elems(imc.A_l, A_l, tags2remove)
        imc.b_u = replace_elems(imc.b_u, b_u, tags2remove)
        imc.b_l = replace_elems(imc.b_l, b_l, tags2remove)

        ## -- Overwrite old attributes ---------------------------------------------------------------------------------
        imc.trans_bounds = trans_bounds_2be_refined


def replace_elems(object2change, elem, tags2remove):
    object2change = np.delete(object2change, tags2remove, axis=0)
    return np.insert(object2change, -1, elem, axis=0)


class IMC:
    def __init__(self, tag):
        self.tag = tag


def bound_transition_kernel(A_u, A_l, b_u, b_l, vertices, centers, rectangles, grid_sorting, std: np.array):
    H = np.concatenate((np.matmul(vertices, np.moveaxis(A_u, 2, 1)) + np.repeat(b_u[:, np.newaxis], axis=1,
                                                                                repeats=vertices.shape[1]),
                        np.matmul(vertices, np.moveaxis(A_l, 2, 1)) + np.repeat(b_l[:, np.newaxis], axis=1,
                                                                                repeats=vertices.shape[1])), axis=1)

    # Overapproximation One-step reachable polytope
    H_rects = np.concatenate((np.min(H, axis=1)[:, :, np.newaxis], np.max(H, axis=1)[:, :, np.newaxis]), axis=2)
    H_rects = np.round(H_rects, 3)

    grid_masks = group_grid_wrt_rectangles(H_rects, grid_sorting)

    trans_bounds = get_trans_bounds(H, H_rects, centers, rectangles, grid_masks, std, use_H=True)
    trans_bounds = trans_bounds.detach().numpy()

    # TEMP
    H_u = np.matmul(vertices, np.moveaxis(A_u, 2, 1)) + np.repeat(b_u[:, np.newaxis], axis=1,
                                                                  repeats=vertices.shape[1])
    H_l = np.matmul(vertices, np.moveaxis(A_l, 2, 1)) + np.repeat(b_l[:, np.newaxis], axis=1,
                                                                  repeats=vertices.shape[1])
    TEST_CROWN_PERF = {'H_u': H_u, 'H_l': H_l, 'H_rects': H_rects}
    return trans_bounds


def get_trans_bounds(H: np.array, H_rects: np.array, grid_centers: np.array, grid_rects: np.array, grid_masks: dict,
                     std: np.array, use_H: bool = False): # \TODO rewrite using einsum
    # TEMP TRANSFORM
    H_rects = torch.from_numpy(H_rects).float()
    grid_centers = torch.from_numpy(grid_centers).float()
    grid_rects = torch.from_numpy(grid_rects).float()
    #

    n = H_rects.shape[1]
    groups = list(itertools.product([0, 1, 2], repeat=n))
    nr_H_rects = H_rects.shape[0]
    nr_grid_rects = grid_centers.shape[0]
    std_torch = torch.from_numpy(std)

    # find locations ----------------------------------------------------------------------------------------------
    max_locs = torch.zeros((nr_H_rects, nr_grid_rects, n))
    min_locs = torch.zeros((nr_H_rects, nr_grid_rects, n))

    grid_rects_tiled = torch.tile(H_rects[:, None], (1, nr_grid_rects, 1, 1))

    # overlap group (2,2,2,..) ------------------------------------------------------------------------------------
    overlap_group_index = groups.index(tuple(2 for _ in range(0, n)))
    overlap_group_tag = tuple(2 for _ in range(0, n))

    # by grouping, maximum at centrum of overlapping groups
    max_locs[grid_masks[overlap_group_tag].row, grid_masks[overlap_group_tag].col] = grid_centers[
        grid_masks[overlap_group_tag].col]

    overlap_group_mask = grid_masks[overlap_group_tag]
    H_rects_centers = torch.tile(torch.mean(H_rects, 2)[:, None], (1, nr_grid_rects, 1))
    if use_H:
        # minimum - if using H
        H = torch.from_numpy(H).float()
        min_locs_overlap = H[overlap_group_mask.row]

        min_lower_overlap = torch.erf(torch.divide(min_locs_overlap - torch.tile(
            grid_rects[overlap_group_mask.col][:, :, 0][:, None], (1, H.shape[1], 1)), std_torch * np.sqrt(2)))
        min_upper_overlap = torch.erf(torch.divide(min_locs_overlap - torch.tile(
            grid_rects[overlap_group_mask.col][:, :, 1][:, None], (1, H.shape[1], 1)), std_torch * np.sqrt(2)))

        trans_bounds_min_overlap = (1 / 2 ** n) * torch.prod(min_lower_overlap - min_upper_overlap, 2)
        min_locs[overlap_group_mask.row, overlap_group_mask.col] = H[
            overlap_group_mask.row, trans_bounds_min_overlap.argmin(dim=1)]
    else:
        # minimum - if using H_rects
        centers_comparison = grid_centers[
                                 None] < H_rects_centers  # \TODO can be done more efficient! by only checking relevant rows/cols

        # convert to np.array due to: https://github.com/pytorch/pytorch/issues/45125
        overlap_group_mask_dims = np.array(centers_comparison[overlap_group_mask.row, overlap_group_mask.col])
        for dim in range(0, n):
            maskDim = overlap_group_mask_dims[:, dim]
            min_locs[overlap_group_mask.row[maskDim], overlap_group_mask.col[maskDim], dim] = \
                H_rects[overlap_group_mask.row[maskDim], dim, 1]
            min_locs[overlap_group_mask.row[~maskDim], overlap_group_mask.col[~maskDim], dim] = \
                H_rects[overlap_group_mask.row[~maskDim], dim, 0]

    # partly overlapping groups (.,2,.,., ..) ---------------------------------------------------------------------
    partly_indices = np.where(np.array([2 in elem for elem in groups if elem != groups[overlap_group_index]]))[0]
    partly_tags = [elem for elem in groups if 2 in elem and elem != groups[overlap_group_index]]

    mask_max_use0 = grid_centers[None] < grid_rects_tiled[:, :, :, 0]
    mask_max_use1 = grid_centers[None] > grid_rects_tiled[:, :, :, 1]
    mask_max_use_center = ~torch.logical_xor(mask_max_use1, mask_max_use0)
    mask_min_use1 = grid_centers[np.newaxis] < H_rects_centers

    for partly_index, partly_tag in zip(partly_indices, partly_tags):
        loc2s = np.where(np.array(groups[partly_index]) == 2)[0]
        loc_non2s = [i for i in range(0, n) if i not in loc2s]
        group_mask = grid_masks[partly_tag]

        # non-overlapping dimensions
        mask_non2s_max = np.array(groups[partly_index])
        mask_non2s_min = 1 - mask_non2s_max
        for loc_non2 in loc_non2s:
            max_locs[group_mask.row, group_mask.col, loc_non2] = \
                H_rects[group_mask.row][:, loc_non2, mask_non2s_max[loc_non2]]
            min_locs[group_mask.row, group_mask.col, loc_non2] = \
                H_rects[group_mask.row][:, loc_non2, mask_non2s_min[loc_non2]]

        # overlapping dimensions
        for loc2 in loc2s:
            mask_max_2s_use0 = np.array(mask_max_use0[group_mask.row, group_mask.col, loc2])
            if np.sum(mask_max_2s_use0) > 0:
                max_locs[group_mask.row[mask_max_2s_use0], group_mask.col[mask_max_2s_use0], loc2] = \
                    H_rects[group_mask.row[mask_max_2s_use0], loc2, 0]

            mask_max_2s_use1 = np.array(mask_max_use1[group_mask.row, group_mask.col, loc2])
            if np.sum(mask_max_2s_use1) > 0:
                max_locs[group_mask.row[mask_max_2s_use1], group_mask.col[mask_max_2s_use1], loc2] = \
                    H_rects[group_mask.row[mask_max_2s_use1], loc2, 1]

            mask_max_2s_use_center = np.array(mask_max_use_center[group_mask.row, group_mask.col, loc2])
            if np.sum(mask_max_2s_use_center) > 0:
                max_locs[group_mask.row[mask_max_2s_use_center], group_mask.col[mask_max_2s_use_center], loc2] = \
                    grid_centers[group_mask.col[mask_max_2s_use_center], loc2]

            mask_min_2s_use1 = np.array(mask_min_use1[group_mask.row, group_mask.col, loc2])
            min_locs[group_mask.row[mask_min_2s_use1], group_mask.col[mask_min_2s_use1], loc2] = \
                H_rects[group_mask.row[mask_min_2s_use1], loc2, 1]

            mask_min_2s_use0 = np.array(~mask_min_2s_use1)
            min_locs[group_mask.row[mask_min_2s_use0], group_mask.col[mask_min_2s_use0], loc2] = \
                H_rects[group_mask.row[mask_min_2s_use0], loc2, 0]

    # non overlapping ---------------------------------------------------------------------------------------------
    non_overlapping_indices = np.setxor1d(np.arange(0, len(groups)),
                                          np.append(partly_indices, np.array(overlap_group_index)))
    non_overlapping_tags = [elem for elem in groups if 2 not in elem]

    i_mask = np.arange(0, n)
    for non_overlapping_index, non_overlapping_tag in zip(non_overlapping_indices, non_overlapping_tags):
        group_mask = grid_masks[non_overlapping_tag]
        max_mask = np.array(groups[non_overlapping_index])
        min_mask = 1 - max_mask
        max_locs[group_mask.row, group_mask.col] = H_rects[group_mask.row][:, i_mask, max_mask]
        min_locs[group_mask.row, group_mask.col] = H_rects[group_mask.row][:, i_mask, min_mask]

    # -- Evaluate gaussian ----------------------------------------------------------------------------------------
    min_lower = torch.erf(torch.divide(min_locs - grid_rects[:, :, 0], std_torch * np.sqrt(2)))
    min_upper = torch.erf(torch.divide(min_locs - grid_rects[:, :, 1], std_torch * np.sqrt(2)))
    max_lower = torch.erf(torch.divide(max_locs - grid_rects[:, :, 0], std_torch * np.sqrt(2)))
    max_upper = torch.erf(torch.divide(max_locs - grid_rects[:, :, 1], std_torch * np.sqrt(2)))

    trans_bounds_min = (1 / 2 ** n) * torch.prod(min_lower - min_upper, 2)
    trans_bounds_min[torch.isnan(trans_bounds_min)] = 0.
    trans_bounds_max = (1 / 2 ** n) * torch.prod(max_lower - max_upper, 2)
    trans_bounds_max[torch.isnan(trans_bounds_max)] = 0.
    trans_bounds = torch.cat((trans_bounds_min[:, :, None], trans_bounds_max[:, :, None]), dim=2)

    return trans_bounds


def get_explicit_bounds(crown_bounds: dict, centers: np.array):
    A_U = np.zeros((centers.shape[0], centers.shape[1], centers.shape[1]))
    A_L = np.zeros((centers.shape[0], centers.shape[1], centers.shape[1]))
    b_U = np.zeros((centers.shape[0], centers.shape[1]))
    b_L = np.zeros((centers.shape[0], centers.shape[1]))

    # A2fill = np.zeros((centers.shape[1], centers.shape[1]))[np.newaxis]
    # A2fill[:, :, :] = np.nan
    # b2fill = np.zeros(centers.shape[1])[np.newaxis]
    # b2fill[:, :] = np.nan

    for count, center in enumerate(centers):
        if tuple(center) in crown_bounds:
            A_U[count] = crown_bounds[tuple(center)]['A_U']
            A_L[count] = crown_bounds[tuple(center)]['A_L']
            b_U[count] = crown_bounds[tuple(center)]['b_U']
            b_L[count] = crown_bounds[tuple(center)]['b_L']
        else:
            raise ValueError('No CROWN bounds given for rectangle: ({}:{}) -> used non-informative bounds'.format(
                count, center))
    return A_U, A_L, b_U, b_L
