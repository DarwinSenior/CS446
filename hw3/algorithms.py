import numpy as np


def accuracy(param, x, y):
    return sum(y == predict(param, x)) * 1.0 / y.size


def predict(param, x):
    w, theta = param
    y = np.dot(x, w) + theta
    return np.sign(y)


def prod(param, x):
    w, theta = param
    return np.dot(x, w) + theta


class Classifier(object):

    def __init__(self, w, theta):
        self.w0 = w
        self.theta0 = theta
        self.reset()

    def reset(self):
        self.w = self.w0
        self.theta = self.theta0
        self.t = 0
        self.errors = []

    def predict(self, x):
        return predict((self.w, self.theta), x)

    def accuracy(self, x, y):
        return accuracy((self.w, self.theta), x, y)

    def name(self):
        return "Generic Algorithm"

    def isMistake(self, x, y):
        return True

    def update(self, x, y):
        pass

    def train(self, x, y):
        self.t += 1
        if (self.isMistake(x, y)):
            self.update(x, y)
            self.errors.append(self.t)

    def trainAll(self, X, Y):
        for i in range(Y.size):
            self.train(X[i, :], Y[i])

    def accumulative(self, N=None):
        N = N or self.t
        X = np.zeros(N)
        i = 0
        n = len(self.errors) - 1
        for k in range(N):
            X[k] = i
            if self.errors[i] == k:
                i = min(i + 1, n)
        return X


class Perceptron(Classifier):

    def __init__(self, w, theta, eta):
        Classifier.__init__(self, w, theta)
        self.eta = eta

    def update(self, x, y):
        self.theta += self.eta * y
        self.w += self.eta * y * x

    def isMistake(self, x, y):
        return (x.dot(self.w) + self.theta) * y <= 0

    def name(self):
        return "perceptron eta=%f" % self.eta


class PerceptronWithMargin(Classifier):

    def __init__(self, w, theta, eta, gamma):
        Classifier.__init__(self, w, theta)
        self.eta = eta
        self.gamma = gamma

    def update(self, x, y):
        self.theta += self.eta * y
        self.w += self.eta * y * x

    def isMistake(self, x, y):
        return (x.dot(self.w) + self.theta) * y < self.gamma

    def name(self):
        return "perceptron eta=%f gamma=%f" % (self.eta, self.gamma)


class Winnow(Classifier):

    def __init__(self, w, theta, alpha):
        Classifier.__init__(self, w, theta)
        self.alpha = alpha

    def update(self, x, y):
        self.w *= self.alpha ** (y * x)

    def isMistake(self, x, y):
        return (x.dot(self.w) + self.theta) * y <= 0

    def name(self):
        return "winnow alpha=%f" % self.alpha


class WinnowWithMargin(Classifier):

    def __init__(self, w, theta, alpha, gamma):
        Classifier.__init__(self, w, theta)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, x, y):
        self.w *= self.alpha ** (y * x)

    def isMistake(self, x, y):
        return (x.dot(self.w) + self.theta) * y < self.gamma

    def name(self):
        return "winnow alpha=%f gamma=%f" % (self.alpha, self.gamma)


class Adagrad(Classifier):

    def __init__(self, w, theta, eta):
        Classifier.__init__(self, w, theta)
        self.eta = eta
        self.G_theta = 0
        self.G_w = np.zeros(self.w.shape)

    def update(self, x, y):
        g_w = - y * x
        g_theta = - y
        self.G_w += g_w**2
        self.G_theta += g_theta**2
        nonzeros = self.G_w != 0
        self.w[nonzeros] += (self.eta * y * x)[nonzeros] / \
            np.sqrt(self.G_w)[nonzeros]
        self.theta += self.eta * y / np.sqrt(self.G_theta)

    def isMistake(self, x, y):
        return (x.dot(self.w) + self.theta) * y <= 0

    def name(self):
        return "adagrad eta=%s" % self.eta


def sampleData(x, y, prop=0.1):
    z = np.arange(0, y.size - 1)
    np.random.shuffle(z)
    sample_size = np.int(y.size * prop)
    train = z[:sample_size]
    test = z[sample_size:sample_size * 2]
    return x[train, :], y[train], x[test], y[test]


def getBestAlgorithm(xtrain, ytrain, xtest, ytest, classifiers, iters=20):
    for clf in classifiers:
        for i in range(iters):
            clf.trainAll(xtrain, ytrain)
        # print('train clf %s done' % (clf.name()))
    return max(classifiers, key=lambda c: c.accuracy(xtest, ytest))


def trainTilConverge(xtrain, ytrain, clf, itermax=10, R=1000):
    index = np.arange(0, ytrain.size - 1)
    for i in range(itermax):
        np.random.shuffle(index)
        for k in index:
            clf.train(xtrain[k, :], ytrain[k])
            if (len(clf.errors) > 0 and clf.t - clf.errors[-1] >= R):
                break
            if (len(clf.errors) > 0 and clf.t - clf.errors[-1] >= R):
                break


def getBestConvergence(xtrain, ytrain, xtest, ytest,
                       classifiers, iters=10, R=1000):
    for clf in classifiers:
        trainTilConverge(xtrain, ytrain, clf, iters, R)
        print('train clf %s done' % (clf.name()))
    return max(classifiers,
               key=lambda c:
               c.accuracy(xtest, ytest) * (c.t - c.errors[-1] >= R))

def classifer_group(n):
    perceptrons = [Perceptron(w=np.zeros(n), theta=0.0, eta=1.0)]
    etas = [1.5, 0.25, 0.03, 0.005, 0.001]
    perceptronMs = [PerceptronWithMargin(w=np.zeros(
        n), theta=0.0, eta=eta, gamma=1.0) for eta in etas]
    alphas = [1.1, 1.01, 1.005, 1.0005, 1.0001]
    winnows = [Winnow(w=np.ones(n), theta=-n*1.0, alpha=alpha) for alpha in alphas]
    gammas = [2.0, 0.3, 0.04, 0.006, 0.001]
    winnowMs = [WinnowWithMargin(w=np.ones(n), theta=-n, alpha=alpha, gamma=gamma)
                for alpha in alphas for gamma in gammas]
    adagrads = [Adagrad(w=np.zeros(n), theta=0.0, eta=eta) for eta in etas]
    return [perceptrons, perceptronMs, winnows, winnowMs, adagrads]
