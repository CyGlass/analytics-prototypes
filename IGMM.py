# First 350 lines or so are functions
# Next 50 lines are for calling functions and plotting results
# Last 100 lines are for comparing the results to those from batch gmm using a new dataset

import numpy as np
import random as rnd
import scipy.spatial.distance as distance
import math
import scipy.stats as stats
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plot
import scipy
import random
import itertools
import traceback
import matplotlib as mpl
import time

# function to calculate mahalanobis distance
def myMahalanobis(x, mu, alpha):
    try:
        return abs(np.matmul((x - mu).transpose(),np.matmul(alpha, x - mu)))
    except ValueError:
        print("alpha = {}, x = {}, mu = {}".format(alpha, x, mu))

# create a new component (representative of a multivariate gaussian distribution)
def create(accumulators, ages, alphas, comProbs, covDets, K, mus, sigmaInv, x):
    accumulators[K] = 1
    ages[K] = 1
    alphas[K] = sigmaInv * np.identity(len(alphas[0]))
    comProbs[K] = 1 / sum(accumulators.values())
    covDets[K] = np.linalg.det(alphas[K]) ** -1
    mus[K] = 0 + x
    K += 1
    return K


# update a component j using the posterior probabilities of data point x
# Algorithm 2 update
def update(accumulators, ages, alphas, comProbs, covDets, D, j, K, mus, printing, x):
    dist = (myMahalanobis(x, mus[j], alphas[j]))

    pxj = 1 / ((2 * math.pi) ** (D / 2) * math.sqrt(covDets[j]))
    pxj *= math.exp(-0.5 * dist)  #

    if printing:
        print(pxj)

    pjx = pxj * comProbs[j]  #
    totProbs = 0
    for i in range(K):
        pxi = 1 / ((2 * math.pi) ** (D / 2) * math.sqrt(covDets[i]))
        pxi *= math.exp(-0.5 * dist)  #
        totProbs += pxi * comProbs[i]  #

    if totProbs == 0:
        print(comProbs)
        print()
    try:
        pjx /= totProbs  #

    except ZeroDivisionError as err:
        print(str(err) + traceback.format_exc() + "pjx = {}, totProbs = {}".format(pjx, totProbs))

    if printing:
        print(pjx)

    ages[j] += 1
    accumulators[j] += pjx

    ej = x - mus[j]
    weight = pjx / accumulators[j]

    deltaMu = weight * ej
    mus[j] += deltaMu

    oldAlpha = alphas[j]

    inputLen = len(ej)
    e = np.ndarray([inputLen, 1])
    for i in range(inputLen):
        e[i, 0] = ej[i]
    ej = e

    newAlpha = np.matmul(oldAlpha, np.matmul(ej, np.matmul(np.transpose(ej), oldAlpha)))
    newAlpha *= (weight * (1 - 3 * weight + weight ** 2)) / ((weight - 1) ** 2 * (weight ** 2 - 2 * weight - 1))
    alphas[j] = newAlpha + oldAlpha / (1 - weight)

    comProbs[j] = accumulators[j] / sum(accumulators.values())

    covDets[j] = (1 - weight) ** D * covDets[j] * (
                1 + (weight * (1 + weight * (weight - 3))) / (1 - weight) * np.matmul(ej.transpose(),
                                                                                      np.matmul(oldAlpha, ej)))

    return mus


# remove a component after it was merged with another component in the model
def remove(accumulators, ages, alphas, comProbs, correspondingData, covDets, j, mus, remainingK):
    for i in range(j, len(ages) - 1):
        accumulators[i] = accumulators.pop(i + 1)
        ages[i] = ages.pop(i + 1)
        alphas[i] = alphas.pop(i + 1)
        comProbs[i] = comProbs.pop(i + 1)
        correspondingData[i] = correspondingData.pop(i + 1)
        covDets[i] = covDets.pop(i + 1)
        mus[i] = mus.pop(i + 1)

    return removeHelper(comProbs, accumulators)
def removeHelper(comProbs, accumulators):
    val = len(accumulators)
    for i in range(val):
        summed = 0
        for j in range(val):
            summed += accumulators[j]
        comProbs[i] = accumulators[i] / summed
    return comProbs


# generate data from 5 normal gaussian distributions
# set plotting to True to see the scatter plots of generated data
def generateData3(plotting=False):
    D = 4;
    K = 0;
    Beta = 0.05;
    delta = 0.01
    vmin = 5;
    spmin = 3
    mus = np.ndarray([D + 1, 2]);
    sigmas = np.ndarray([D + 1, 2, 2])
    mus = np.array([[1, 1], [10, 10], [10, 1], [1, 10], [5, 5]])

    for i in range(D + 1):
        sigmas[i] = [[1, 0], [0, 1]]

    startingMus = mus
    numvectors = 1500;
    lenvectors = 2
    X = np.ndarray((numvectors, lenvectors))
    startingCorrespondences = []

    maxZ = 0
    count = 0
    for i in range(numvectors):
        distant = False
        index = random.randint(0, 4)
        startingCorrespondences.append(index)
        X[i] = np.random.multivariate_normal(mus[index], sigmas[index])
        for j in range(lenvectors):
            ZScore = (X[i][j] - mus[index][j]) / sigmas[index][j, j]
            if abs(ZScore) > 3:
                distant = True
            if abs(ZScore) > abs(maxZ):
                maxZ = ZScore
        if distant:
            count += 1
    if plotting:
        splot = plot.subplot(1, 1, 1)

        Xs = [[], [], [], [], []]
        Ys = [[], [], [], [], []]
        colors = ['r', 'b', 'k', 'y', 'm']

        for i in range(len(X)):
            index = startingCorrespondences[i]
            Xs[index].append(X[i, 0])
            Ys[index].append(X[i, 1])
        for i in range(5):
            plot.scatter(Xs[i], Ys[i], c=colors[i])
            v, w = np.linalg.eigh(sigmas[i])
            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180.0 * angle / np.pi  # convert to degree
            Xs_to_plot = Xs[i]
            Ys_to_plot = Ys[i]
            plot.scatter(Xs_to_plot, Ys_to_plot, c=colors[i])
            plot.scatter((mus[i][0]), (mus[i][1]), c='g')

            ell = mpl.patches.Ellipse(mus[i], v[0], v[1], 180.0 + angle, color=colors[i % 5])
            ell.set_clip_box(splot.bbox)

            ell.set_alpha(0.5)
            splot.add_artist(ell)

    return [X, startingMus, sigmas, startingCorrespondences, maxZ, count / numvectors]
# return the generated data, and the mus and sigmas of the gaussians used to generate the data


# simulate the IGMM process for input data stream X
# set printing parameter to true to see final correspondences from the IGMM generated mus to the mus used in data generation
def simulate(X, sigmas, startingMus, delta=1, dimension=2, vmin=5, spmin=3, printing=False):
    sigma = np.std(X[:100])  # sigma should be representative of only a part of the data stream
    sigmaInv = sigma ** -1
    alpha = sigmaInv * np.identity(dimension)
    covarDet = np.linalg.det(np.linalg.inv(alpha))
    mDistMax = 2  # 5 * sigma
    D = dimension

    alphas = {}
    mus = {}
    covDets = {}
    comProbs = {}
    ages = {}
    accumulators = {}
    correspondingData = {}
    ZDict = {}

    alphas[0] = alpha
    covDets[0] = covarDet
    mus[0] = X[0]
    comProbs[0] = 1
    ages[0] = 1
    accumulators[0] = 1
    K = 1
    remainingK = len(startingMus) + 4
    correspondingData[0] = [0]
    correspondingData[-1] = []
    correspondingData[-2] = []

    for i in range(len(X)):
        ZDict[i] = []
        if i == 0:
            continue
        updated = False

        for j in range(K):
            if j >= K:
                break


        for j in range(K):
            if i == j:
                continue

            dist = (myMahalanobis(X[i], mus[j], alphas[j]))
            if printing:
                print('(' + str(i) + ', ' + str(j) + ')')
                print(dist)
            if dist < mDistMax:
                if printing:
                    print('call update')
                    print()
                correspondingData[j].append(i)
                ZDict[i].append(getZScores(X[i], mus[j], np.linalg.inv(alphas[j]), dist))
                mus = update(accumulators, ages, alphas, comProbs, covDets, D, j, K, mus, printing, X[i])
                # print('mu3 = {}'.format(mus[j]))
                # print()
                updated = True

        if not updated and remainingK == 0:
            correspondingData[-1].append(i)

        if not updated and not remainingK == 0:
            if printing:
                print('call create')
                print()
            correspondingData[K] = [i]
            K = create(accumulators, ages, alphas, comProbs, covDets, K, mus, sigmaInv, X[i])
            remainingK -= 1

    oldZDict = {}
    # if correspondingData[-1]:
    #    [alphas, correspondingData, mus, K, comProbs, covDets, ZDict, oldZDict] = simulateMature(accumulators, ages, alphas, comProbs, correspondingData, covDets, D, K, mus, X[correspondingData[-1]], ZDict, mDistMax = 20)

    if printing:
        print(K)
    norms = np.ndarray([D + 1, K])
    difference = 0
    correspondences = np.ndarray(D + 1)
    for i in range(D + 1):  # D + 1 = len(startingMus)
        for j in range(K):
            norms[i, j] = np.linalg.norm(startingMus[i] - mus[j])
        index = np.argmin(norms[i])
        difference += norms[i, index]
        correspondences[i] = index

        if printing:
            print("mu = {}, sigma = {}".format(mus[index], np.linalg.inv(alphas[index])))
            print("starting mu = {}, starting Sigma = {}".format(startingMus[i], sigmas[i]))
            print("corresponding data = {}\n".format(correspondingData[index]))

    if printing:
        print(str(difference) + '\n')
        print(correspondingData)

    K = mergeClusters(accumulators, ages, alphas, comProbs, correspondingData, covDets, D, K, mDistMax, mus, remainingK,
                      sigmaInv, X, ZDict)
    return [alphas, correspondingData, difference, mus, K, comProbs, covDets, ZDict, oldZDict, accumulators]

# function to do a merge pass after convergence
def mergeClusters(accumulators, ages, alphas, comProbs, correspondingData, covDets, D, K, mDistMax, mus, remainingK,
                  sigmaInv, X, ZDict):
    doneMerging = False
    maxDist = mDistMax * 1.5
    while not doneMerging:
        doneMerging = True
        for i in range(K):
            if i >= K:
                break

            for j in range(i):
                if j >= K or i >= K:
                    break
                dist1 = myMahalanobis(mus[i], mus[j], alphas[j])
                dist2 = myMahalanobis(mus[j], mus[i], alphas[i])
                dist = (dist1 + dist2) / 2
                if dist < maxDist:
                    doneMerging = False
                    points = correspondingData[j]  # look at all points in j
                    for x in correspondingData[i]:
                        if x not in points:
                            points.append(x)  # look at all points in i not in j

                    K = mergeClustersHelper(accumulators, ages, alphas, comProbs, correspondingData, covDets, D, i, j,
                                            K, mDistMax, points, mus, remainingK, sigmaInv, X, ZDict)
                    remainingK += 1

    return K


# function to merge two clusters into one new cluster
def mergeClustersHelper(accumulators, ages, alphas, comProbs, correspondingData, covDets, D, j1, j2, K, mDistMax,
                        mergeX, mus, remainingK, sigmaInv, X, ZDict):
    comProbs = remove(accumulators, ages, alphas, comProbs, correspondingData, covDets, j1, mus, remainingK)
    K -= 1;
    remainingK += 1
    comProbs = remove(accumulators, ages, alphas, comProbs, correspondingData, covDets, j2, mus, remainingK)
    K -= 1;
    remainingK += 1
    correspondingData[K] = [mergeX[0]]
    K = create(accumulators, ages, alphas, comProbs, covDets, K, mus, sigmaInv, X[mergeX[0]])

    for i in range(1, len(mergeX)):
        dist = myMahalanobis(X[mergeX[i]], mus[K - 1], alphas[K - 1])
        if dist < mDistMax:
            correspondingData[K - 1].append(mergeX[i])
            # ZDict[i].append(getZScores(X[mergeX[i]], mus[K - 1], np.linalg.inv(alphas[K - 1]), dist))
            mus = update(accumulators, ages, alphas, comProbs, covDets, D, K - 1, K, mus, False, X[mergeX[i]])

        else:
            correspondingData[-1].append(mergeX[i])

    return K


def getSoftProbs(alphas, comProbs, covDets, K, mus, Xini):
    X = np.copy(Xini)
    D = K - 1
    softProbs = np.ndarray([len(Xini), K])
    probs = np.ndarray([K])
    for i in range(len(Xini)):
        for j in range(K):
            dist = (myMahalanobis(X[i], mus[j], alphas[j]))
            probs[j] = 1 / ((2 * math.pi) ** (D / 2) * math.sqrt(covDets[j])) * math.exp(-0.5 * dist) * comProbs[j]

        probsum = sum(probs)

        for j in range(K):
            try:
                softProbs[i, j] = math.exp(math.log(probs[j]) - math.log(probsum))
            except ValueError:
                softProbs[i, j] = 0
    return softProbs

def getZScores(x, mu, sigma, dist):
    ZScores = [dist]
    for i in range(len(x)):
        ZScores.append(((x[i] - mu[i]) / sigma[i,i]))
    return ZScores

Data = generateData3(True) #generate some data without plotting it
Xini = np.copy(Data[0]) #save a copy of the input datastream (avoid losing it during update)

[alphas, correspondingData, difference, mus, K, comProbs, covDets, ZDict, oldZDict, accumulators] = simulate(X = Data[0], sigmas = Data[2], startingMus = Data[1], printing = False)

newSigmas = []
for i in range(len(alphas)):
    newSigmas.append(np.linalg.inv(alphas[i]))
M = GaussianMixture(n_components = 5, covariance_type = 'full').fit(np.copy(Xini))

colors = ['r', 'b', 'y', 'k', 'm', 'c']
for i in range(len(mus)):
    plot.figure(1)
    A = correspondingData[i]

    Xs_to_plot = Xini[A, 0]
    Ys_to_plot = Xini[A, 1]
    plot.scatter(Xs_to_plot, Ys_to_plot, c=colors[i % 6])
    plot.scatter((mus[i][0]), (mus[i][1]), c='g')
for i in range(K):
    #print(len(correspondingData[i]))
    #print(comProbs[i])
    print("the {}th mu = {}".format(i,mus[i]))
    print("the {}th sigma = {}\n".format(i,np.linalg.inv(alphas[i])))


#Here is some code to evaluate the efficacy of the igmm based on the batch model
matchedMus = {}
newAlphas = {}
newComProbs = {}
newCovDets = {}

for i in range(len(mus)):
    distances = []
    for j in range(len(M.means_)):
        distances.append(np.linalg.norm(M.means_[j] - mus[i]))
    matchedMus[distances.index(min(distances))] = mus[i]
    newAlphas[distances.index(min(distances))] = alphas[i]
    newComProbs[distances.index(min(distances))] = comProbs[i]
    newCovDets[distances.index(min(distances))] = covDets[i]

newSigmas = {}
for i in range(len(newAlphas)):
    newSigmas[i] = np.linalg.inv(newAlphas[i])

newData = generateData3(False)  # generate some data without plotting i
Xnew = newData[0]
softProbs = getSoftProbs(newAlphas, newComProbs, newCovDets, K, matchedMus, Xnew)
# print(softProbs)
sPArr = []
for i in range(len(softProbs)):
    sPArr.append(np.argmax(softProbs[i]))

# print(sPArr)
batch = M.predict(Xnew)
# print(batch)

startingMus = newData[1]
startingSigmas = newData[2]
startingCorrespondences = newData[3]

matchedStartingSigmas = {}
matchedStartingMus = {}
componentMap = {}
for i in range(len(startingMus)):
    distances = []
    for j in range(len(M.means_)):
        distances.append(np.linalg.norm(startingMus[i] - M.means_[j]))
    matchedStartingMus[distances.index(min(distances))] = startingMus[i]
    matchedStartingSigmas[distances.index(min(distances))] = startingSigmas[i]
    componentMap[i] = (distances.index(min(distances)))
startingMus = matchedStartingMus
startingSigmas = matchedStartingSigmas

for i in range(len(mus)):
    print("{}th starting mu = {}".format(i, startingMus[i]))
    print("{}th incremental mu = {}".format(i, matchedMus[i]))
    print("{}th batch mu = {}".format(i, M.means_[i]))
    print()
    print("{}th starting sigma = {}".format(i, startingSigmas[i]))
    print("{}th incremental sigma = {}".format(i, newSigmas[i]))
    print("{}th batch sigma = {}".format(i, M.covariances_[i]))
    print("\n\n")

for i in range(len(startingCorrespondences)):
    startingCorrespondences[i] = componentMap[startingCorrespondences[i]]

# print(startingCorrespondences)
bCount = 0
iCount = 0
for i in range(len(sPArr)):
    if sPArr[i] == startingCorrespondences[i]:
        iCount += 1
    if batch[i] == startingCorrespondences[i]:
        bCount += 1
iAcc = iCount / len(startingCorrespondences) * 100
bAcc = bCount / len(startingCorrespondences) * 100
print("The IGMM was {}% accurate.".format(iAcc))
print("The batch GMM was {}% accurate.".format(bAcc))
