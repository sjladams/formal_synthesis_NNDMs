from imdp import IMDP
from synthesis import Synthesis
from support import generate_discretization, import_model, merge_q_yes_q_no_regions, get_crown_bounds
from parameters import dx, SYSTEM_TYPES, VERF_USE_LTL, VERF_P, VERF_SPEC_TYPE, VERF_K

if __name__ == '__main__':
    rectangles, centers = generate_discretization(dx, add_full=True)

    models = {system_type: import_model(system_type, plot=True) for system_type in SYSTEM_TYPES}

    crown_bounds_all = get_crown_bounds(models, rectangles, centers)
    imdp = IMDP(rectangles, centers, crown_bounds_all)

    synthesis = Synthesis(imdp, spec_type=VERF_SPEC_TYPE, k=VERF_K, p=VERF_P, use_LTL=VERF_USE_LTL)

    tags2remove, new_rectangles, new_centers = merge_q_yes_q_no_regions(imdp.rectangles, synthesis)
    crown_bounds_all = get_crown_bounds(models, new_rectangles, new_centers, crown_bounds_all)
    imdp.merge(tags2remove, new_rectangles, new_centers, crown_bounds_all)
    synthesis.update_grid_elements()

    synthesis.run_synthesis()

    synthesis.plot(models, plot_type='lower bounds', mark_des=True, plot_des_tags=True, mark_obs=True)
    synthesis.plot(models, plot_type='classification', mark_des=True, plot_des_tags=True, mark_obs=True)


