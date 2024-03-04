import torch
from torch.utils.data import Dataset
from torch import nn
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from bound_propagation import BoundModelFactory
import functools

DIRPATH = os.path.dirname(__file__)


class CustomDataset(Dataset):
    def __init__(self, x_in, x_out):
        self.data = (torch.from_numpy(x_in).float(), torch.from_numpy(x_out).float())

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]


class Network(nn.Sequential):
    def __init__(self, nr_layers: int, nr_neurons: int, act_func: str, n: int, **kwargs):
        if nr_layers > 1:
            if act_func == 'relu':
                func = nn.ReLU
            elif act_func == 'sigmoid':
                func = nn.Sigmoid
            else:
                raise NotImplementedError

            hidden_structure = [[func(), nn.Linear(nr_neurons, nr_neurons)] for _ in range(1, nr_layers)]
            structure = [nn.Linear(n, nr_neurons)] + [elem for sublist in hidden_structure for elem in sublist] + \
                        [func(), nn.Linear(nr_neurons, n)]
        else:
            structure = [nn.Linear(n, n)]
        super().__init__(*structure)


def load_model(model_path: str, system: 'ImportSystem', plot: bool = False, **kwargs):
    model = Network(**kwargs)
    model.load_state_dict(torch.load(model_path))
    factory = BoundModelFactory()
    model = factory.build(model)
    model.eval()
    if plot:
        plot_org_sys_phase_portrait(system, **kwargs)
        plot_learned_sys_phase_portrait(model, **kwargs)
    return model


def train_model(model_path: str, system: 'ImportSystem', train_by_keras: bool, plot: bool = False, **kwargs):
    x_data, y_data = generate_data(system, **kwargs)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    plot_org_sys_phase_portrait(system, **kwargs)

    if train_by_keras:
        keras_model = keras_train_nn(x_train, y_train, x_test, y_test, **kwargs)
        pytorch_model = keras2pytorch_model(keras_model, **kwargs)
    else:
        pytorch_model = pytorch_train_nn(x_train, y_train, x_test, y_test, **kwargs)

    if plot:
        plot_learned_sys_phase_portrait(pytorch_model, **kwargs)

    torch.save(pytorch_model.state_dict(), f"{model_path}.pt")
    factory = BoundModelFactory()
    pytorch_model = factory.build(pytorch_model)
    pytorch_model.eval()
    return pytorch_model


def build_keras_nn_model(n: int, nr_layers: int, nr_neurons: int, act_func: str, **kwargs) \
        -> tf.keras.models.Model:
    if nr_layers > 1:
        structure = [tf.keras.layers.Dense(nr_neurons, activation=act_func, name="dense_0",
                                           input_shape=(n,))] + \
                    [tf.keras.layers.Dense(nr_neurons, activation=act_func, name="dense_{}".format(i)) for i in
                     range(1, nr_layers)] + [tf.keras.layers.Dense(n, name="predictions")]
    else:
        structure = [tf.keras.layers.Dense(n, name="predictions", input_shape=(n,))]

    model = tf.keras.models.Sequential(structure)
    model.summary()
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
    opt = tf.keras.optimizers.Adamax(learning_rate=0.01)
    nn_model.compile(loss='mean_squared_error',
                     optimizer=opt,
                     metrics=['accuracy'])
    return myCallback


def keras_train_nn(x_train: np.array, y_train: np.array, x_test, y_test: np.array, epochs: int, **kwargs):
    # build the model
    nn_model = build_keras_nn_model(**kwargs)

    # train the model
    callback_class_instance = keras_create_callback(nn_model)
    callbacks = callback_class_instance()
    history = nn_model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, shuffle=True, callbacks=[callbacks])
    nn_model.evaluate(x_test, y_test, verbose=2)

    plot_training_loss(history.history['loss'], history.history['val_loss'])
    return nn_model


def keras2pytorch_model(keras_model, **kwargs):
    parameters = keras_model.get_weights()
    model = Network(**kwargs)
    for i, _ in enumerate(parameters):
        if i % 2 == 0:
            model[int(i / 2) * 2].weight.data = torch.from_numpy(np.transpose(parameters[i]))
        else:
            model[int(i / 2) * 2].bias.data = torch.from_numpy(np.transpose(parameters[i]))
    return model


def pytorch_train_nn(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, epochs: int, **kwargs):
    train_ds = CustomDataset(x_train, y_train)
    train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    test_ds = CustomDataset(x_test, y_test)
    test_ds_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    model = Network(**kwargs)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    valid_losses = []

    for epoch in range(1, epochs + 1):
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


def generate_data(system: 'ImportSystem', n: int, ss: np.array, train_dataset_size: int, **kwargs):
    x_in = {i: (ss[i, 1] - ss[i, 0]) * np.random.random_sample(train_dataset_size) + ss[i, 0] for i in range(0, n)}
    to_pass = [x_in[dim] for dim in x_in]
    x_out = {i: np.array(list(map(functools.partial(system.generator, dim=i), *to_pass))) for i in range(0, n)}
    x_data = np.array(list(x_in.values())).T
    y_data = np.array(list(x_out.values())).T
    return x_data, y_data


def plot_org_sys_phase_portrait(system: 'ImportSystem', ss: np.array, dx: np.array, n: int, **kwargs):
    lins = [np.arange(ss[dim, 0], ss[dim, 1] + 0.5 * dx[dim], dx[dim]) for dim in range(0, n)]
    xs = np.meshgrid(*lins)
    As = [np.array(list(map(functools.partial(system.generator, dim=dim), *xs))) for dim in range(0, n)]

    if n == 2:
        _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
        _ax.quiver(*(xs + [AsElem - xsElem for AsElem, xsElem in zip(As, xs)]), scale=1, units='x')
        _ax.set_title('Original System - Scale 1')
        plt.show()

        # _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
        # _ax.quiver(*(xs+[AsElem - xsElem for AsElem, xsElem in zip(As,xs)]))
        # _ax.set_title('scale auto')
        # plt.show()
    if n == 3:
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


def plot_learned_sys_phase_portrait(trained_nn_model, n: int, ss: np.array, dx: np.array, **kwargs):
    lins = []
    for dim in range(0, n):
        lins += [np.arange(ss[dim, 0], ss[dim, 1] + 0.5 * dx[dim], dx[dim])]

    Xs = np.meshgrid(*lins)

    As = [np.zeros(Xs[dim].shape) for dim in range(0, n)]

    for idx, pair in np.ndenumerate(np.array(Xs[0])):
        Ipair = np.array([Xs[dim][idx] for dim in range(0, len(Xs))])
        Ipair = torch.from_numpy(Ipair).float()
        op = trained_nn_model(Ipair)
        # op = trained_nn_model.predict(np.array([Ipair]))[0]
        # print('idx {}, pair {}'.format(idx, pair))
        for idy, elem in enumerate(op):
            As[idy][idx] = elem

    if n == 2:
        _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(10, 10))
        arrows = (Xs + [AsElem - XsElem for AsElem, XsElem in zip(As, Xs)])
        _ax.quiver(*(Xs + [AsElem - XsElem for AsElem, XsElem in zip(As, Xs)]), scale=1, units='x')
        _ax.set_title('Learned System - Scale 1')
        plt.show()
    if n == 3:
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


def plot_training_loss(train_losses, valid_losses):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='train')
    ax.plot(valid_losses, label='test')
    ax.set_yscale("log")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()
