from algorithms import sampleData, getBestConvergence, classifer_group
from gen import gen
import numpy as np
from matplotlib import pylab

grid = np.zeros((5, 5))
for k,n in enumerate([40, 80, 120, 160, 200]):
    y, x = gen(l=10, m=20, n=n, number_of_instances=50000, noise=0)
    xtrain, ytrain, xtest, ytest = sampleData(x, y)
    classifierss = classifer_group(x.shape[1])
    trained_clfs = [getBestConvergence(
        xtrain, ytrain, xtest, ytest, clfs
    ) for clfs in classifierss]
    print("n = %d" % n)
    for (i, clf) in enumerate(trained_clfs):
        print("%s => %f" % (clf.name(), clf.accuracy(xtest, ytest)))
        grid[k, i] = clf.t - clf.errors[-1]

pylab.ion()
pylab.figure("Mistake vs n plot")
pylab.legend(handles=[pylab.plot([40, 80, 120, 160, 200], grid[i, :], label=clf.__class__.__name__)[0]
                      for (i, clf) in enumerate(trained_clfs)
])
pylab.savefig("mistake-n.png")
input()
