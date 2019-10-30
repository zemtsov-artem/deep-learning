import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from scipy.special import xlogy


def stable_softmax(x):
    tmp = x - x.max(axis=1, keepdims=True)
    np.exp(tmp, out=x)
    x /= x.sum(axis=1, keepdims=True)
    return x

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    out = x[:]
    out[out < 0] = 0
    out[out > 0] = 1
    return out


def crossentropy_loss(y_true, y_prob):
    return - xlogy(y_true, y_prob).sum()


class DNNClassifier():
    def __init__(self, hidden_layer_sizes=(100,),
                 activation_functions=(None, None),
                 loss_function=crossentropy_loss,
                 batch_size=1, learning_rate=0.001,
                 max_iter=200, random_state=42, verbose=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        self._label_binarizer = LabelBinarizer()

    def __forward_layer(self, x, w, activation_function):
        out = np.dot(x, w)
        return activation_function(out)

    def __forward_propagate(self, x):
        activations = [x]
        for next_layer, activation in zip(self.weights, self.activation_functions):
            out = self.__forward_layer(activations[-1], next_layer, activation)
            activations.append(out)

        return activations

    def __back_propagation(self, activations, y):
        weights = self.weights
        gradients = [np.empty_like(layer) for layer in weights]

        deltas1 = activations[2] - y # 256,10
        gradients[1] = np.dot(activations[1].T, deltas1) # 128,10

        deltas2 = np.dot(deltas1, weights[1].T) * relu_derivative(activations[1]) # 256,128

        gradients[0] = np.dot(activations[0].T, deltas2) # hotim 784,128

        return gradients

    def __init_layer(self, input_size, output_size):
        a = 2.0 / (input_size + output_size)
        w = np.random.uniform(-a, a, (input_size, output_size))
        return w

    def fit(self, X, y):
        np.random.seed(self.random_state)

        y_train = y
        X_train = X
        y = self._label_binarizer.fit_transform(y)
        num_classes = len(self._label_binarizer.classes_)

        n, p = X.shape
        s = self.hidden_layer_sizes[0]

        self.weights = [
            self.__init_layer(p, s),
            self.__init_layer(s, num_classes)
        ]
        for j in range(self.max_iter):
            accumulated_loss = 0.0

            for i in range(0, n, self.batch_size):
                X_batch = X[i: i + self.batch_size]
                y_batch = y[i: i + self.batch_size]

                activations = self.__forward_propagate(X_batch)

                y_prob = activations[-1]

                accumulated_loss += self.loss_function(y_batch, y_prob)
                gradients = self.__back_propagation(activations, y_batch)
                gradients = [gradient / self.batch_size for gradient in gradients]
                self.weights = [weight - self.learning_rate * grad for weight, grad in
                                zip(self.weights, gradients)]

            loss = accumulated_loss / X.shape[0]
            y_pred = self.predict(X_train)
            accuracy = (y_pred == y_train).mean()
            print("Epoch {}/{};\t Train accuracy: {:.3f} \t Loss : {:.3f}".format(j + 1, self.max_iter, accuracy,
                                                                                      loss))

        return self

    def predict(self, X):
        activations = self.__forward_propagate(X)
        y_pred = activations[-1]
        return self._label_binarizer.inverse_transform(y_pred)


import tensorflow as tf

if __name__ == '__main__':
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    
    
    x_train /= 255
    x_test /= 255

    activation_functions = (relu, stable_softmax)

    estimator = DNNClassifier(hidden_layer_sizes=(300,),
                              activation_functions=activation_functions,
                              batch_size=8,
                              learning_rate=0.1,
                              max_iter=20,
                              random_state=66)
    estimator.fit(x_train, y_train)

    y_pred = estimator.predict(x_test)
    print("Accuracy on test dataset: %s " % (y_pred == y_test).mean())
