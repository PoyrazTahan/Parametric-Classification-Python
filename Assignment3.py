from __future__ import division

import math

import matplotlib as mlt
import numpy as np
import pandas as pd
import numpy
from numpy.linalg import inv

mlt.use('TkAgg')
import matplotlib.pyplot as plt
from random import *


class DataAll:
    """
    Holds all of the data
    holds properties of the whole data
    """

    def __init__(self, data):
        # raw data
        self.dataSetDF = data
        # what are the class names
        self.labels = set(self.dataSetDF['c3'])
        # How many class are there?
        self.numberOfLabels = len(self.labels)
        # classified data array, every object will be a dataSub
        self.subDF = []

        for i in self.labels:
            # creates a dataSub class for each class
            temp = self.dataSetDF[self.dataSetDF['c3'] == i]
            temp = temp.drop('c3', 1)
            x = DataSub(temp, str(i), self)
            self.subDF.append(x)

        ### gets the max and min values of the whole data for plotting purposes
        self.maxX = self.dataSetDF.iloc[0][0]
        self.minX = self.dataSetDF.iloc[0][0]
        self.maxY = self.dataSetDF.iloc[0][1]
        self.minY = self.dataSetDF.iloc[0][1]

        for i in range(len(self.dataSetDF)):
            # print self.dataSetDF.iloc[i][0]
            if self.dataSetDF.iloc[i][0] > self.maxX:
                self.maxX = self.dataSetDF.iloc[i][0]
            if self.dataSetDF.iloc[i][0] < self.minX:
                self.minX = self.dataSetDF.iloc[i][0]

        for i in range(len(self.dataSetDF)):
            # print self.dataSetDF.iloc[i][1]
            if self.dataSetDF.iloc[i][1] > self.maxY:
                self.maxY = self.dataSetDF.iloc[i][1]
            if self.dataSetDF.iloc[i][1] < self.minY:
                self.minY = self.dataSetDF.iloc[i][1]
        ### max min deal end

        print self.maxX
        print self.maxY
        print self.minX
        print self.minX

class DataSub:
    """
    Holds classified data (each class has its own object)
    holds properties of the classified data
    """

    def __init__(self, data, label, prt):
        self.parent = prt
        self.dataSet = data
        self.dataLen = len(self.dataSet[0:])
        self.label = label
        self.splitPnt = self.dataLen / 2

        self.subTest = DataTest(self.dataSet[int(self.splitPnt):], self.label, self)

        self.subTrain = DataTrain(self.dataSet[:int(self.splitPnt)], self.label, self)

        self.probDF = pd.DataFrame()


class DataTrain:
    """
    Holds train part of the classified data, there is one of this class for each DataSub
    holds properties of the trainSet data
    """

    def __init__(self, data, lbl, prt):
        self.parent = prt
        self.dataSet = data
        self.dataLen = len(self.dataSet[0:])
        self.label = lbl

        self.meanX = self.dataSet["c1"].mean()
        self.meanY = self.dataSet["c2"].mean()

        self.means = [self.meanX, self.meanY]

        self.dfMean = pd.DataFrame(data=self.means)
        # make your own cov method
        self.covMat = self.dataSet.cov()
        # self.covMat = pd.DataFrame(data = covMatrix(self))

        self.prior = self.dataLen / self.parent.dataLen


class DataTest:
    """
    Holds test part of the classified data ,there is one of this class for each DataSub
    holds properties of the testSet data
    """

    def __init__(self, data, lbl, prt):
        self.parent = prt
        self.dataSet = data
        self.dataLen = len(self.dataSet[0:])
        self.label = lbl


def Main():
    '''
    ... creates an object with layers of data sets
    ... puts it to mathematical analyzing and classifies it and then gives the accuracy.
    ... then plots it. It also plots the area of influence by each set. !!!But it takes a while to bit map.

    :return:
    '''

    # the directory that contains the data
    directory = "./data/two_class/data5.txt"
    data = getDataSetTable(directory)  # reading the data
    dfAll = pd.DataFrame(data=data)

    dfAll.columns = ["c{}".format(i + 1) for i in range(3)]

    d1 = DataAll(dfAll)

    analytics(d1, directory)

    plotty1(d1)


def getDataSetTable(path):
    '''
    the function thas reads the data from txt file it splits according to space

    :param path:
    :return:

    '''
    table = []
    with open(path, 'r') as file:
        for line in file:
            tmp = []
            for word in line.split():
                tmp.append(float(word.strip()))
            table.append(tmp)
    return table


def probCals(x, mean, cov):
    """

    :param x: is the x y point that in being considered
    :param mean: is the mean vector size : 1 X N (N being the # parameter) of the distribution that is being considered
    :param cov: is the covariance Matrix size : N X N 1XN
                (N being the # parameter) of the distribution that is being considered
    :return: the gausien probability
    """
    d = len(x)
    difVec = pd.DataFrame(x.subtract(mean))

    covInv = cov.as_matrix()
    covDet = np.linalg.det(covInv)
    covInv = inv(covInv)
    covInv = pd.DataFrame(data=covInv)
    difVecT = difVec.transpose()

    exp = math.exp((-1 / 2) * difVecT.dot(covInv).dot(difVec).values)

    return (1 / (math.pow(2 * math.pi, d / 2) * math.pow(covDet, 1 / 2)) * exp)


def gFunc(prob, prior):
    """

    :param prob: takes the gausian probability
    :param prior: with its corresponding sets prior value = P(n_j)/P(N)
    :return: the g value of the function // the end probability
    """
    return math.log1p(prob) + math.log1p(prior)


def analytics(d1, directory):
    """

    :param d1: the data object that contains the whole data // this is since we will use both test and train
    :param directory: For the output to look nice
    :return: prints the accuracy
    """
    errCnt = 0
    crrCnt = 0
    for k in range(len(d1.subDF)):
        print "Class:", str(d1.subDF[k].subTrain.label), "Mean Vector:\n", d1.subDF[k].subTrain.dfMean, "\n"

        print "Class:", str(d1.subDF[k].subTrain.label), "Covarience Matrix:\n", d1.subDF[k].subTrain.covMat, "\n"

        for j in range(len(d1.subDF[k].subTest.dataSet)):
            maxIdx = 0
            maxG = 0
            x = d1.subDF[k].subTest.dataSet.iloc[j][0]
            y = d1.subDF[k].subTest.dataSet.iloc[j][1]
            ans = d1.subDF[k].subTest.label
            temp = [x, y]
            X = pd.DataFrame(data=temp)

            for i in range(len(d1.subDF)):
                p = probCals(X, d1.subDF[i].subTrain.dfMean, d1.subDF[i].subTrain.covMat)
                g = gFunc(p, d1.subDF[i].subTrain.prior)
                if g > maxG:
                    maxIdx = i
                    maxG = g
            pred = str(d1.subDF[maxIdx].label)
            if pred == ans:
                crrCnt += 1
            else:
                errCnt += 1

    totalCnt = crrCnt + errCnt

    print "Data:", directory, "> Accuracy (%):", crrCnt / totalCnt * 100, "[Errors: ", errCnt, "/", totalCnt, "]"


def covMatrix(dataClass):
    """

    :param dataClass: takes the train set and calculate its covariance matrix N X N
    :return:
    """
    df = dataClass.dataSet.as_matrix()
    length = len(df)
    mean = dataClass.dfMean.as_matrix()

    for i in range(length):
        for j in range(len(df[0])):
            df[i][j] = df[i][j] - mean[j]

    dfT = df.T

    return numpy.dot(dfT, df) / length




######### PLOTTING FUNCTIONS ##########


def plotty1(data):
    """
    plots the whole data set acording to their class
    shades the area of influence
    :param data:
    :return:
    """

    for i in range(data.numberOfLabels):
        plt.scatter(data.subDF[i].dataSet['c1'], data.subDF[i].dataSet['c2'], 0.15)

    plt.show()

    paintProbArea(data)


def paintProbArea(d1):
    colorArray = []

    # to get a better shaded area increase the incrementNumber
    incrementNumber = 100

    for i in range(d1.numberOfLabels):
        r = int(random() * 255)
        g = int(random() * 255)
        b = int(random() * 255)
        colorArray.append([r, g, b])


    x = np.linspace(d1.minX - 5, d1.maxX + 5, incrementNumber)
    y = np.linspace(d1.minY - 5, d1.maxY + 5, incrementNumber)

    predGroupX = []
    predGroupY = []
    for i in range(d1.numberOfLabels):
        predGroupX.append([])
        predGroupY.append([])

    for i in range(len(x)):
        for j in range(len(y)):
            pred = prediction(x[i], y[j], d1)
            predGroupX[pred].append(x[i])
            predGroupY[pred].append(y[j])

    for i in range(d1.numberOfLabels):
        plt.scatter(predGroupX[i], predGroupY[i], 0.7)

    plt.show()


def prediction(x, y, d1):
    '''

    :param x: x coordinate of the point that you want to make a prediction for
    :param y: y coordinate of the point that you want to make a prediction for
    :param d1:
    :return:
    '''
    maxG = 0
    temp = [x, y]
    X = pd.DataFrame(data=temp)
    for i in range(d1.numberOfLabels):
        p = probCals(X, d1.subDF[i].subTrain.dfMean, d1.subDF[i].subTrain.covMat)
        g = gFunc(p, d1.subDF[i].subTrain.prior)
        if g > maxG:
            maxIdx = i
            maxG = g

    return maxIdx




if __name__ == '__main__':
    Main()
