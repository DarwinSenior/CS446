from algorithms import sampleData, getBestAlgorithm, classifer_group
from gen import gen

# experiment 3

for m in [100, 500, 1000]:
    ytrain, xtrain = gen(l=10, m=m, n=1000, number_of_instances=50000, noise=True)
    ytest, xtest = gen(l=10, m=m, n=1000, number_of_instances=10000, noise=False)
    xtrain_t, ytrain_t, xtest_t, ytest_t = sampleData(xtrain, ytrain)
    classifierss = classifer_group(xtrain.shape[1])
    trained_clfs = [getBestAlgorithm(
        xtrain_t, ytrain_t, xtest_t, ytest_t, clfs, iters=20) for clfs in classifierss]
    print("for m=%d"%m)
    for clf in trained_clfs:
        clf.reset()
        for i in range(20):
            clf.trainAll(xtrain, ytrain)
        print(clf.name())
        print(clf.accuracy(xtest, ytest))
        print(clf.accuracy(xtrain, ytrain))
