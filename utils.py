from argparse import ArgumentParser
import json
import numpy as np


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--debug', type=eval, default=True, help='debug')
    parser.add_argument('--system_name', type=str, choices=['linear-slow-center', 'linear-fast-center', 'jackson',
                                                       'jackson-nl', 'jackson-nl-alt', 'nonlin-2d-lin-2d', 'car',
                                                       'car-simple', 'car-3d', 'car-2d', 'unstable-nl',
                                                       'cartpole-stabilizer'],
                        default='linear-fast-center', help='system tag')
    parser.add_argument('--modes', type=list, default=[1], help='list of the number of modes (provided as integers)')
    parser.add_argument('--spec_name', type=str, choices=['default', '2d-center', 'jackson', 'jackson-nl',
                                                          'jackson-nl-compl', 'car-straight', 'car-corner',
                                                          'car-overtake', 'car-turn', 'nonlin-2d-lin-2d'],
                        default='2d-center', help='safety specification')
    parser.add_argument('--data_setting', type=int, default=0)
    parser.add_argument('--load', type=eval, default=True, help='load models if available')
    parser.add_argument('--parallelize', type=eval, default=False, help='parallelize computations')
    parser.add_argument('--nr_pools', type=int, default=4, help='number of pools used for parallization')
    parser.add_argument('--use_cpp', type=eval, default=False, help='use cpp implementation for value iteration')
    return parser.parse_args()

def load_json(filename: str):
    with open(f"configs/{filename}.json", "r") as read_file:
        data = json.load(read_file)
    return data


def param_handler(param_name: str, system_name: str = None, setting_tag: int = None):
    if system_name is None:
        params = load_json(param_name)
        if setting_tag is None:
            return params
        else:
            return params[str(setting_tag)]
    else:
        params = load_json(param_name)[system_name]
        if setting_tag is None:
            return params
        else:
            if "basis" in params:
                params_basis = params['basis']
                if "options" in params:
                    params_option = params['options'][str(setting_tag)]
                    return {**params_basis, **params_option}
                else:
                    return params_basis
            else:
                return params[str(setting_tag)]


def load_params(args):
    bnn_params = param_handler('bnn', args.system_name)
    data_params = param_handler('data', args.system_name, args.data_setting)
    synth_params = param_handler('synthesis', args.system_name)
    spec_params = {'spec': param_handler('specification', args.spec_name)}

    params = {**bnn_params, **data_params, **synth_params, **spec_params, **vars(args)}
    return check_params(params)


def process_spec(specs: dict | list, n: int):
    if isinstance(specs, list):
        specs = np.array(specs, dtype=np.float64)
        if not specs.shape[-1] == n:
            raise ValueError
        else:
            return specs
    else:
        for key in specs.keys():
            specs[key] = process_spec(specs[key], n=n)
        return specs


def check_params(params):
    if params['spec_name'] == 'jackson-nl-compl':
        params['use_ltlf'] = True
    else:
        params['use_ltlf'] = False

    params['n'] = len(params['dx']) # number of dimensions system

    # set list types to np.arrays & check dims
    keys2check = ['dx', 'std', 'ss']
    for key in keys2check:
        params[key] = np.array(params[key], dtype=np.float64)
        if not params[key].shape[-1] == params['n']:
            raise ValueError(f"param: {key} should be of dims {params['n']}")

    params['spec'] = process_spec(params['spec'], params['n'])

    # construct systems
    params['systems'] = [f"{params['system_name']}-mode{mode_id}" for mode_id in params['modes']]

    return params


