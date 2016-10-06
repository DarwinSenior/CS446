from algorithms import Adagrad
from gen import gen
import numpy as np

(y, x) = gen(l=10, m=20, n=40, number_of_instances=10000, noise=1)

def trainAdagrad(clf, x=x, y=y, rounds=50):
    errors = np.zeros(rounds)
    hingeloss = np.zeros(rounds)
    for round in rounds:
        clf.trainAll(x, y)
        errors[round] = 1-clf.accuracy(x, y)
        hingeloss[round] = 1-clf.hingeLoss(x, y)
