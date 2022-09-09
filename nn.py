import torch
from torch.utils.data import Dataset
from torch import nn
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from bound_propagation import crown, crown_ibp, ibp
import functools

from parameters import DIMS, NUM_TRAIN_POINTS, dx, STATE_SPACE, MAX_EPOCHS, NEURONS, HIDDEN_LAYERS, ACT_FUNC_TYPE

DIRPATH = os.path.dirname(__file__)
kerasTrain = True


@crown
@crown_ibp
@ibp
class Network(nn.Sequential):
    def __init__(self, *args):
        if args:
            # To support __get_index__ of nn.Sequential when slice indexing
            # CROWN (and implicitly CROWN-IBP) is doing this underlying
            super().__init__(*args)
        else:
            if HIDDEN_LAYERS > 1:
                if ACT_FUNC_TYPE == 'ReLU':
                    hidden_structure = [[nn.ReLU(), nn.Linear(NEURONS, NEURONS)] for _ in range(1, HIDDEN_LAYERS)]
                    structure = [nn.Linear(DIMS, NEURONS)] + [elem for sublist in hidden_structure for elem in sublist] + \
                                [nn.ReLU(), nn.Linear(NEURONS, DIMS)]
                elif ACT_FUNC_TYPE == 'Sigmoid':
                    hidden_structure = [[nn.Sigmoid(), nn.Linear(NEURONS, NEURONS)] for _ in range(1, HIDDEN_LAYERS)]
                    structure = [nn.Linear(DIMS, NEURONS)] + [elem for sublist in hidden_structure for elem in sublist] + \
                                [nn.Sigmoid(), nn.Linear(NEURONS, DIMS)]
            else:
                structure = [nn.Linear(DIMS, DIMS)]
            super().__init__(*structure)


def generate_data(system: 'ImportSystem'):
    x_in = dict()
    for dim in range(0, DIMS):
        x_in[dim] = (STATE_SPACE[dim, 1] - STATE_SPACE[dim, 0]) * np.random.random_sample(NUM_TRAIN_POINTS) + \
                    STATE_SPACE[dim, 0]

    to_pass = [x_in[dim] for dim in x_in]
    x_out = dict()
    for dim in range(0, DIMS):
        x_out[dim] = np.array(list(map(functools.partial(system.generator, dim=dim), *to_pass)))

    x_data = np.array(list(x_in.values())).T
    y_data = np.array(list(x_out.values())).T

    return x_data, y_data


def plot_org_sys_phase_portrait(system: 'ImportSystem'):
    lins = [np.arange(STATE_SPACE[dim, 0], STATE_SPACE[dim, 1] + 0.5 * dx[dim], dx[dim]) for dim in range(0, DIMS)]
    xs = np.meshgrid(*lins)
    As = [np.array(list(map(functools.partial(system.generator, dim=dim), *xs))) for dim in range(0, DIMS)]

    if DIMS == 2:
        _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
        _ax.quiver(*(xs + [AsElem - xsElem for AsElem, xsElem in zip(As, xs)]), scale=1, units='x')
        _ax.set_title('Original System - Scale 1')
        plt.show()

        # _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
        # _ax.quiver(*(xs+[AsElem - xsElem for AsElem, xsElem in zip(As,xs)]))
        # _ax.set_title('scale auto')
        # plt.show()
    if DIMS == 3:
        # plot 2D for each level of phi
        # arrows = (xs+[AsElem - xsElem for AsElem, xsElem in zip(As,xs)])
        # for phi in range(0,arrows[0].shape[2]):
        #     _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
        #     _ax.quiver(*[arrows[0][:,:,phi], arrows[1][:,:,phi], arrows[3][:,:,phi], arrows[4][:,:,phi]], scale=1,units='x')
        #     _ax.set_xlabel('x_1')
        #     _ax.set_ylabel('x_2')
        #     _ax.set_title('Original - Scale 1 - x3: {}'.format(lins[2][phi]))
        #     plt.show()

        _ax = plt.figure().add_subplot(projection='3d')
        _ax.quiver(*(xs + [AsElem - xsElem for AsElem, xsElem in zip(As, xs)]))
        _ax.set_xlabel('x')
        _ax.set_ylabel('y')
        _ax.set_zlabel('phi')
        _ax.view_init(elev=10., azim=0)
        _ax.set_title('Original System')
        plt.show()

        # for x in range(0,arrows[0].shape[0]):
        #     _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
        #     _ax.quiver(*[arrows[1][x,:,:], arrows[2][x,:,:], arrows[4][x,:,:], arrows[5][x,:,:]], scale=1,units='x')
        #     _ax.set_xlabel('x_2')
        #     _ax.set_ylabel('x_3')
        #     _ax.set_title('Original - Scale 1 - x1: {}'.format(lins[0][x]))
        #     _ax.set_xlim(x-1,x+1)
        #     plt.show()
        #
        # for y in range(0,arrows[0].shape[2]):
        #     _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
        #     _ax.quiver(*[arrows[0][:,y,:], arrows[2][:,y,:], arrows[3][:,y,:], arrows[5][:,y,:]], scale=1,units='x')
        #     _ax.set_xlabel('x_1')
        #     _ax.set_ylabel('x_3')
        #     _ax.set_title('Original - Scale 1 - x3: {}'.format(lins[1][y]))
        #     _ax.set_xlim(x-1,x+1)
        #     plt.show()

        # _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
        # _ax.quiver(*[arrows[0][:,:,phi], arrows[1][:,:,phi], arrows[3][:,:,phi], arrows[4][:,:,phi]])
        # _ax.set_xlabel('x_1')
        # _ax.set_ylabel('x_2')
        # _ax.set_title('Original - Auto Scale- x3: {}'.format(lins[2][phi]))
        # plt.show()


def plot_learned_sys_phase_portrait(trained_nn_model):
    lins = []
    for dim in range(0, DIMS):
        lins += [np.arange(STATE_SPACE[dim, 0], STATE_SPACE[dim, 1] + 0.5 * dx[dim], dx[dim])]

    Xs = np.meshgrid(*lins)

    As = [np.zeros(Xs[dim].shape) for dim in range(0, DIMS)]

    for idx, pair in np.ndenumerate(np.array(Xs[0])):
        Ipair = np.array([Xs[dim][idx] for dim in range(0, len(Xs))])
        Ipair = torch.from_numpy(Ipair).float()
        op = trained_nn_model(Ipair)
        # op = trained_nn_model.predict(np.array([Ipair]))[0]
        # print('idx {}, pair {}'.format(idx, pair))
        for idy, elem in enumerate(op):
            As[idy][idx] = elem

    if DIMS == 2:
        _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
        arrows = (Xs + [AsElem - XsElem for AsElem, XsElem in zip(As, Xs)])
        _ax.quiver(*(Xs + [AsElem - XsElem for AsElem, XsElem in zip(As, Xs)]), scale=1, units='x')
        _ax.set_title('Learned System - Scale 1')
        plt.show()
    if DIMS == 3:
        arrows = (Xs + [AsElem - XsElem for AsElem, XsElem in zip(As, Xs)])

        # plot 2D for each level of phi
        for phi in range(0, arrows[0].shape[2]):
            _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
            _ax.quiver(*[arrows[0][:, :, phi], arrows[1][:, :, phi], arrows[3][:, :, phi], arrows[4][:, :, phi]],
                       scale=1, units='x')
            _ax.set_xlabel('x_1')
            _ax.set_ylabel('x_2')
            _ax.set_title('Learned System - Scale 1 - x3: {}'.format(lins[2][phi]))
            plt.show()

            # _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
            # _ax.quiver(*[arrows[0][:,:,phi], arrows[1][:,:,phi], arrows[3][:,:,phi], arrows[4][:,:,phi]])
            # _ax.set_xlabel('x_1')
            # _ax.set_ylabel('x_2')
            # _ax.set_title('Learned - Auto Scale - x3: {}'.format(lins[2][phi]))
            # plt.show()


def train_model(model_path: str, system: 'ImportSystem', plot: bool = False):
    x_data, y_data = generate_data(system)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    plot_org_sys_phase_portrait(system)

    if kerasTrain:
        keras_model = keras_train_nn(x_train, y_train, x_test, y_test)
        pytorch_model = keras2pytorch_model(keras_model)
    else:
        pytorch_model = pytorch_train_nn(x_train, y_train, x_test, y_test)

    if plot:
        plot_learned_sys_phase_portrait(pytorch_model)

    # Check if the models/ directory exist, if not create it.
    model_dir = DIRPATH + "/models"
    check_model_dir = os.path.isdir(model_dir)
    if not check_model_dir:
        os.makedirs(model_dir)

    torch.save(pytorch_model.state_dict(), model_path)
    return pytorch_model


def load_model(model_path: str, system: 'ImportSystem', plot: bool = False):
    model = Network()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if plot:
        plot_org_sys_phase_portrait(system)
        plot_learned_sys_phase_portrait(model)
    return model


def build_keras_nn_model() -> tf.keras.models.Model:
    if HIDDEN_LAYERS > 1:
        if ACT_FUNC_TYPE == 'ReLU':
            structure = [tf.keras.layers.Dense(NEURONS, activation='relu', name="dense_0", input_shape=(DIMS,))] + \
                        [tf.keras.layers.Dense(NEURONS, activation='relu', name="dense_{}".format(i)) for i in
                         range(1, HIDDEN_LAYERS)] + [tf.keras.layers.Dense(DIMS, name="predictions")]
        elif ACT_FUNC_TYPE == 'Sigmoid':
            structure = [tf.keras.layers.Dense(NEURONS, activation='sigmoid', name="dense_0", input_shape=(DIMS,))] + \
                        [tf.keras.layers.Dense(NEURONS, activation='sigmoid', name="dense_{}".format(i)) for i in
                         range(1, HIDDEN_LAYERS)] + [tf.keras.layers.Dense(DIMS, name="predictions")]
    else:
        structure = [tf.keras.layers.Dense(DIMS, name="predictions", input_shape=(DIMS,))]

    model = tf.keras.models.Sequential(structure)

    # when you use summary the input object is not displayed as it is not a layer.
    model.summary()

    # display the layers
    print(model.layers)

    return model


def keras_create_callback(nn_model):
    VAL_LOSS = 1e-5

    # Implement callback function to stop training
    # when accuracy reaches e.g. ACCURACY_THRESHOLD = 0.95
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_loss') < VAL_LOSS):
                print("\n Reached %2.5f%% validation loss, so stopping training!!" % (VAL_LOSS))
                self.model.stop_training = True

    # assigning an optimizer
    opt = tf.keras.optimizers.Adamax(lr=0.01)
    nn_model.compile(loss='mean_squared_error',
                     optimizer=opt,
                     metrics=['accuracy'])

    return myCallback


def keras_train_nn(x_train: np.array, y_train: np.array, x_test, y_test: np.array):
    # build the model
    nn_model = build_keras_nn_model()

    # train the model
    callback_class_instance = keras_create_callback(nn_model)
    callbacks = callback_class_instance()
    history = nn_model.fit(x_train, y_train, epochs=MAX_EPOCHS, validation_split=0.2, shuffle=True,
                           callbacks=[callbacks])
    nn_model.evaluate(x_test, y_test, verbose=2)

    plot_training_loss(history.history['loss'], history.history['val_loss'])
    return nn_model


def keras2pytorch_model(keras_model):
    parameters = keras_model.get_weights()

    model = Network()

    for i, _ in enumerate(parameters):
        if i % 2 == 0:
            model[int(i / 2) * 2].weight.data = torch.from_numpy(np.transpose(parameters[i]))
        else:
            model[int(i / 2) * 2].bias.data = torch.from_numpy(np.transpose(parameters[i]))
    return model


class CustomDataset(Dataset):
    def __init__(self, xIn, xOut):
        self.data = (torch.from_numpy(xIn).float(), torch.from_numpy(xOut).float())

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]


def pytorch_train_nn(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
    train_ds = CustomDataset(x_train, y_train)
    train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    test_ds = CustomDataset(x_test, y_test)
    test_ds_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    model = Network()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    valid_losses = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for xIn, xOut in train_ds_loader:
            optimizer.zero_grad()
            predict = model(xIn)
            loss = loss_fn(predict, xOut)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xIn.size(0)

        model.eval()
        for xIn, xOut in test_ds_loader:
            predict = model(xIn)
            loss = loss_fn(predict, xOut)

            valid_loss += loss.item() * xIn.size(0)

        train_loss = train_loss / len(train_ds_loader.sampler)
        valid_loss = valid_loss / len(test_ds_loader.sampler)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch:{} Train Loss:{:.4f} valid Losss:{:.6f}'.format(epoch, train_loss, valid_loss))

    plot_training_loss(train_losses, valid_losses)
    return model


def plot_training_loss(train_losses, valid_losses):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='train')
    ax.plot(valid_losses, label='test')
    ax.set_yscale("log")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()
