import utils
from imdp import IMDP
from synthesis import Synthesis
from support import generate_discretization, import_model, merge_q_yes_q_no_regions, get_crown_bounds

if __name__ == '__main__':
    args = utils.parse_arguments()
    params = utils.load_params(args)

    rectangles, centers = generate_discretization(add_full=True, **params)

    models = {system_tag: import_model(system_tag, plot=True, **params) for system_tag in params['systems']}

    crown_bounds_all = get_crown_bounds(models, rectangles, centers)
    imdp = IMDP(rectangles, centers, crown_bounds_all, params['std'])

    synthesis = Synthesis(imdp=imdp, **params)

    tags2remove, new_rectangles, new_centers = merge_q_yes_q_no_regions(rectangles=imdp.rectangles, ss=params['ss'],
                                                                        spec=params['spec'], synthesis=synthesis)
    crown_bounds_all = get_crown_bounds(models, new_rectangles, new_centers, crown_bounds_all)
    imdp.merge(tags2remove, new_rectangles, new_centers, crown_bounds_all)
    synthesis.update_grid_elements()

    synthesis.run_synthesis()

    synthesis.plot(models, ss=params['ss'], plot_type='lower bounds', mark_des=True, plot_des_tags=True, mark_obs=True)
    synthesis.plot(models, ss=params['ss'], plot_type='classification', mark_des=True, plot_des_tags=True, mark_obs=True)


