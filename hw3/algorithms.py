import numpy as np


def correct(param, x, y):
    return sum(y == predict(param, x)) / y.size

def predict(param, x):
    w, theta = param
    y = np.dot(w, x) + theta
    return np.sign(y)


def prod(param, x):
    w, theta = param
    return np.dot(w, x) + theta


def perceptron(eta):
    def update(param, x, y):
        w, theta = param
        yp = predict(param, x)
        pred = predict(param, x) != y
        w += pred * eta * y * x
        theta += pred * eta * y
        return (w, theta)
    return update


def perceptron_with_margin(gamma, eta):
    def update(param, x, y):
        w, theta = param
        pred = prod(param, x) * y < gamma
        w += pred * eta * y * x
        theta += pred * eta * y
        return (w, theta)
    return update


def winnow(alpha):
    def update(param, x, y):
        w, theta = param
        pred = predict(param, x) != y
        w *= alpha**(pred * y * x)
        return (w, theta)
    return update


def winnow_with_margin(gamma, alpha):
    def update(param, x, y):
        w, theta = param
        pred = prod(param, x) * y < gamma
        w *= alpha**(pred * y * x)
        return (w, theta)
    return update


def adagrad(eta, alpha):
    vars = {'g_theta': 0, 'g_w': None}

    def update(param, x, y):
        pred = predict(param, x, y) != y
        w, theta = param
        g_w = - pred * y * x
        g_theta = - pred * y
        vars['g_w'] = (vars['g_w'] or np.zeros(w.shape)) + g_w**2
        vars['g_theta'] += g_theta**2
        w += pred * eta * y * x / vars['g_w']
        theta += pred * eta * y / vars['g_theta']
        return (w, theta)
    return update

def runAlgorithm(X, Y, param0, update):
    w, theta = param0
    param = (w.copy, theta)
    for i in range(Y.size):
        param = update(param, X[i,:], Y[i])
    return param
