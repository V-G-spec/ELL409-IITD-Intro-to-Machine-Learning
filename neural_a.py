import numpy as np
import pandas as pd
import math, time, sys


class NeuralNetwork:
    def __init__(self, X, params):
        self.params = params
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.activation_function = self.get_activation_function(
            params["activation_function"]
        )
        self.loss_func = self.get_loss_function(params["loss_function"])
        self.architecture = params["architecture"]
        self.initialize_weights_and_biases(X)

    def get_activation_function(self, activation_function):
        if activation_function == 0:
            return self.sigmoid
        elif activation_function == 1:
            return self.tanh
        elif activation_function == 2:
            return self.relu

    def get_loss_function(self, loss_function):
        if loss_function == 0:
            return self.cross_entropy
        elif loss_function == 1:
            return self.mean_squared_error

    def sigmoid(self, x, derivative=False):
        if derivative == True:
            return np.exp(-x) / ((1 + np.exp(-x)) ** 2)
        return 1 / (1 + np.exp(-x))

    def tanh(self, x, derivative=False):
        if derivative == True:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)

    def relu(self, x, derivative=False):
        if derivative == True:
            return 1 * (x > 0)
        return np.maximum(0, x)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def cross_entropy(self, y, y_hat):
        return -np.sum(np.sum(y * np.log(y_hat))) / y.shape[0]

    def mean_squared_error(self, y, y_hat):
        return np.mean(np.square(y - y_hat))

    def update_learning_rate(self, eta0, iteration):
        if params["learning_rate_strat"] == 0:
            return eta0
        return eta0 / math.sqrt(iteration)

    def initialize_weights_and_biases(self, X):
        n = X.shape[1]
        for i in range(len(self.architecture)):
            m = self.architecture[i]
            curr = np.float32(
                np.random.normal(size=(n + 1, m)) * math.sqrt(2 / (m + n + 1))
            )
            curr_weights, curr_biases = (
                curr[1:, :],
                curr[0, :],
            )
            curr_biases = curr_biases.reshape(1, curr_biases.shape[0])
            curr_weights, curr_biases = np.float64(curr_weights), np.float64(
                curr_biases
            )
            self.weights.append(curr_weights)
            self.biases.append(curr_biases)
            n = m

    def forward_propagate(self, X):
        self.zs_list = [X]
        self.as_list = [X]
        for i in range(len(self.weights)):
            a = np.dot(self.zs_list[i], self.weights[i]) + self.biases[i]
            self.as_list.append(a)
            if i == len(self.weights) - 1:
                z = self.softmax(a)
                self.zs_list.append(z)
            else:
                z = self.activation_function(a)
                self.zs_list.append(z)
        return self.zs_list[-1]

    def back_propagate(self, X, y):
        self.deltas = []
        y_hat = self.zs_list[-1]
        self.deltas.append((y_hat - y) / y.shape[0])  # y_hat - y
        for i in range(len(self.weights) - 1, -1, -1):
            delta = np.multiply(
                self.deltas[-1].dot(self.weights[i].T),
                self.activation_function(self.as_list[i], derivative=True),
            )
            self.deltas.append(delta)
        self.deltas.reverse()
        return self.deltas

    def update_weights(self, X, y, epoch):
        eta = self.params["initial_learning_rate"]
        self.deltas = self.back_propagate(X, y)[1:]
        for i in range(len(self.weights)):
            grad_weights = np.dot(self.zs_list[i].T, self.deltas[i])
            grad_bias = np.sum(self.deltas[i], axis=0)
            self.weights[i] -= eta * grad_weights
            self.biases[i] -= eta * grad_bias
            eta = self.update_learning_rate(
                self.params["initial_learning_rate"], epoch + 1
            )

    def train(self, X, y):
        for epoch in range(1, params["epochs"] + 1):
            for batch in range(0, X.shape[0], params["batch_size"]):
                X_batch, y_batch = (
                    X[batch : batch + params["batch_size"]],
                    y[batch : batch + params["batch_size"]],
                )
                self.forward_propagate(X_batch)
                self.update_weights(X_batch, y_batch, epoch)
            self.loss_history.append(self.loss_func(y, self.forward_propagate(X)))

    def predict(self, X):
        return self.forward_propagate(X)


def get_params(lines):
    params = {}
    params["epochs"] = int(lines[0].strip())
    params["batch_size"] = int(lines[1].strip())
    params["architecture"] = list(int(i) for i in lines[2].strip()[1:-1].split(","))
    # architecture([100,50,10] implies 2 hidden layers with 100 and 50 neurons and 10 neurons in the output layer
    params["learning_rate_strat"] = int(lines[3].strip())  # 0 for fixed, 1 for adaptive
    params["initial_learning_rate"] = float(lines[4].strip())
    params["activation_function"] = int(
        lines[5].strip()
    )  # 0 for sigmoid, 1 for tanh, 2 for relu
    params["loss_function"] = int(lines[6].strip())  # 0 for CE, 1 for MSE
    params["seed"] = int(lines[7].strip())  # seed for np.random.normal()
    return params


def read_and_encode(input_path):
    df = pd.read_csv(input_path, header=None)
    data = df.to_numpy()
    X, y = data[:, 1:] / 255, data[:, 0]
    y = pd.get_dummies(y, columns=y).to_numpy()
    return X, y


if __name__ == "__main__":
    input_path, output_path = sys.argv[1], sys.argv[2]
    X, y = read_and_encode(input_path + "toy_dataset_train.csv")
    param_file = sys.argv[3]
    lines = open(param_file).readlines()
    params = get_params(lines)
    np.random.seed(params["seed"])

    network = NeuralNetwork(X, params)
    network.train(X, y)
    for i in range(len(network.weights)):
        weight_and_bias = np.concatenate((network.biases[i], network.weights[i]))
        output = output_path + "w_{0}.npy".format(i + 1)
        np.save(output, weight_and_bias)
    predictions = np.argmax(network.predict(X), axis=1)
    np.save(output_path + "predictions.npy", predictions)
