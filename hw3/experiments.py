from algorithms import sampleData, getBestAlgorithm, classifer_group
from gen import gen
from matplotlib import pylab

# experiment 1


y, x = gen(l=10, m=100, n=500, number_of_instances=50000, noise=0)
xtrain, ytrain, xtest, ytest = sampleData(x, y)
classifierss = classifer_group(x.shape[1])
trained_clfs = [getBestAlgorithm(
    xtrain, ytrain, xtest, ytest, clfs, iters=20) for clfs in classifierss]

pylab.figure("l=10, m=100, n=500")
pylab.legend(handles=[pylab.plot(clf.accumulative(),
                                 label=clf.__class__.__name__)[0]
                      for clf in trained_clfs])
pylab.savefig("plot-10-100-500.png")
for clf in trained_clfs:
    print(clf.name())
    print(clf.accuracy(xtest, ytest))

print("next round")


y, x = gen(l=10, m=100, n=1000, number_of_instances=50000, noise=0)
xtrain, ytrain, xtest, ytest = sampleData(x, y)
classifierss = classifer_group(x.shape[1])
pylab.figure("l=10, m=100, n=1000")
trained_clfs = [getBestAlgorithm(
    xtrain, ytrain, xtest, ytest, clfs, iters=20) for clfs in classifierss]
pylab.legend(handles=[pylab.plot(clf.accumulative(),
                                 label=clf.__class__.__name__)[0]
                      for clf in trained_clfs])
pylab.savefig("plot-10-100-1000.png")
for clf in trained_clfs:
    print(clf.name())
    print(clf.accuracy(xtest, ytest))
