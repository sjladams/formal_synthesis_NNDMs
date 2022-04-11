# {nr: {dim: [min, max]
SAFETY_SPECS = {'2d-center': {
    'D': {0: {0: [-0.5, 0.5],
              1: [-0.5, 0.5]}},
    'O': {0: {0: [0.5, 1.0],
              1: [0.5, 1.0]}}
},
    'default': {'D': {0: {0: [-0.5, 0.5],
                          1: [-0.5, 0.5]}},
                'O': {}},
    'jackson': {'D': {0: {0: [-1., 0.],
                          1: [-1., 0.]},
                      1: {0: [0., 1.],
                          1: [0., 1.]}
                      },
                'O': {0: {0: [-2., -1.],
                          1: [-2., -1.]},
                      1: {0: [-1., 0.],
                          1: [0., 1.]},
                      2: {0: [0., 1.],
                          1: [-1., 0.]},
                      3: {0: [1., 2.],
                          1: [1., 2.]}
                      }
                },
    'jackson-nl': {'D': {0: {0: [-0.75, 0.75],
                             1: [-0.75, 0.75]}
                         },
                   'O': {}
                   },
    'jackson-nl-compl': {'D1': {0: {0: [-1.25, 0.],
                                    1: [1., 2.]},  # [1., 1.875]
                                1: {0: [-1.75, 0.],  # [-1.75, -0.125]
                                    1: [-2., -1.5]},
                                2: {0: [1.25, 2.],  # [1.125, 2.]
                                    1: [-1.75, 0.]}},
                         'D2': {0: {0: [-0.75, 0.75],
                                    1: [-0.75, 0.75]}},
                         'O': {0: {0: [-1.75, -1.25],
                                   1: [-0.75, 1.]},
                               1: {0: [-1., -0.5],
                                   1: [-1.25, -0.75]},  # [-1.25, -0.875]
                               2: {0: [0.5, 1.25],  # [0.5, 1.125]
                                   1: [-1.75, -1.25]},
                               3: {0: [0.75, 1.],
                                   1: [-0.5, 0.5]},
                               4: {0: [0.75, 1.75],
                                   1: [0.75, 1.75]}}
                         },
    'car-straight': {'D': {0: {0: [2.5, 4.],
                               1: [1.5, 2.5]}},
                     'O': {0: {0: [2.5, 4.0],
                               1: [0., 1.5]},
                           1: {0: [2.5, 4.],
                               1: [2.5, 4.]}}},
    'car-corner': {'D': {0: {0: [3., 4.],
                             1: [0.5, 1.5]}},
                   'O': {0: {0: [0., 4.],
                             1: [0., 0.5]},
                         1: {0: [3., 4.],
                             1: [1.5, 4.]}}},
    'car-overtake': {'D': {0: {0: [8., 10.],
                               1: [0., 1.]}},
                     'O': {0: {0: [4., 6.],
                               1: [0., 1.]}}},
    'car-turn': {'D': {0: {0: [3., 4.],
                           1: [3., 4.]}},
                 'O': {0: {0: [3., 4.],
                           1: [0., 3.]}}},
    'NonLin2Dlin2D': {'D': {0: {0: [1., 2.],
                                1: [0.5, 2.]}},
                      'O': {0: {0: [0.5, 2.],
                                1: [-0.75, 0.]},
                            1: {0: [-0.75, 0.],
                                1: [0.5, 2.]}}}
}