import numpy as np

# Verification / Synthesis
VERF_K = 20
VERF_P = 0.95
VERF_PROJ_DIMS = [0,1]
VERF_SPEC_TYPE = 'jackson-nl-compl'
if VERF_SPEC_TYPE == 'jackson-nl-compl':
    VERF_USE_LTL = True
else:
    VERF_USE_LTL = False

# NN
NEURONS = 20
HIDDEN_LAYERS = 3
ACT_FUNC_TYPE = 'ReLU' # 'Sigmoid'

MAX_ERF_INPUT = 3.3
LOOKUP_DEC = 3

VERF_USE_CPP = False
VERF = True
LOAD = False
SAVE = True
TRAIN = False
PARL = False
NR_POOLS = 4

SYSTEM_TYPES = ['NonLin2Dlin2D-mode1']
# SYSTEM_TYPES = ['UnstableNL']

## Jackson NL
# SYSTEM_TYPES = ['jackson-nl-mode1', 'jackson-nl-mode2', 'jackson-nl-mode3', 'jackson-nl-mode4']
# SYSTEM_TYPES = ['jackson-nl-mode1']

# ## turning - car
# SYSTEM_TYPES = ['car-3d-mode9','car-3d-mode10',
#                 'car-3d-mode11','car-3d-mode12','car-3d-mode8']

# # high dim
# SYSTEM_TYPES = ['jackson-nl-mode1-Alt', 'jackson-nl-mode2-Alt',
#                 'jackson-nl-mode3-Alt']

# # overtaking - car
# SYSTEM_TYPES = ['car-3d-mode7','car-3d-mode1','car-3d-mode2','car-3d-mode3',
#                 'car-3d-mode4','car-3d-mode5','car-3d-mode6']

DIMS = 2
REFINE = True
VERF_ROUNDS = 2

# ## turn car
# dx = np.array([0.5, 0.5])
# STD = np.array([0.05, 0.05])
# STATE_SPACE = np.array([[0., 4.],
#                         [0., 4.]])

## Jackson nl
dx = np.array([0.25, 0.25])
STD = np.array([0.15, 0.15])
STATE_SPACE = np.array([[-2., 2.],
                        [-2., 2.]])
# STATE_SPACE = np.array([[1., 3.],
#                         [1., 3.]])

# # high dim
# dx = np.array([0.25, 0.25])
# STD = np.array([0.1, 0.1])
# STATE_SPACE = np.array([[-2., 2.],
#                         [-2., 2.]])

# # Overtaking car
# dx = np.array([0.5, 0.5])
# STD = np.array([0.01, 0.01])
# STATE_SPACE = np.array([[0., 10.],
#                         [0., 2.]])

if DIMS == 3:
    ## overtaking - car
    dx = np.append(dx, 0.1)
    STD = np.append(STD, 0.01)
    STATE_SPACE = np.append(STATE_SPACE, [[-0.5, 0.5]], axis=0)

    # # turning - car
    # dx = np.append(dx, 0.125)
    # STD = np.append(STD, 0.01)
    # STATE_SPACE = np.append(STATE_SPACE, [[-0.25, 1.]], axis=0)
elif DIMS == 4:
    ## high dim
    dx = np.append(dx, [0.2,0.2])
    STD = np.append(STD, [0.01,0.01])
    STATE_SPACE = np.append(STATE_SPACE, [[-0.4, 0.4],
                                          [-0.4, 0.4]], axis=0)
elif DIMS == 5:
    ## high dim
    dx = np.append(dx, [0.2,0.2,0.3])
    STD = np.append(STD, [0.01,0.01,0.01])
    STATE_SPACE = np.append(STATE_SPACE, [[-0.3, 0.3],
                                          [-0.3, 0.3],
                                          [-0.3, 0.3]], axis=0)

np.random.seed(2)
NUM_TRAIN_POINTS = 4000
MAX_EPOCHS = 200