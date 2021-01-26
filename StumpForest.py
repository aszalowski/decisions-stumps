from Stump import Stump

import numpy
import collections
import operator

class StumpForest:
    def __init__(self, trainingDataset, numOfStumps=1000, numOfFeatures=2, sampleMethod='undersample'):
        self.stumps = list() 
        self.numOfStumps = numOfStumps
        self.sampleMethod = sampleMethod
        # self.numOfFeatures = numOfFeatures

        # trainingDataset = self.randomOverSampling(trainingDataset)

        self.possibleClasses = self.calculatePossibleClasses(trainingDataset)

        for _ in range(numOfStumps):
            trainingSample = self.sample(trainingDataset)
            newStump = Stump(trainingSample, numOfFeatures, self.possibleClasses)
            self.stumps.append(newStump)

    def calculatePossibleClasses(self, trainingDataset):
        classColumn = trainingDataset[:, -1]
        classes, counts = numpy.unique(classColumn, return_counts=True)
        return dict(zip(classes, counts))

    def predict(self, row):
        predictions = [stump.decide(row) for stump in self.stumps]
        return max(set(predictions), key=predictions.count)

    def sample(self, trainingDataset):
        if isinstance(self.sampleMethod, float):
            return self.bootstrapSampling(trainingDataset)
        elif isinstance(self.sampleMethod, str):
            if self.sampleMethod == 'oversample':
                return self.randomOverSampling(trainingDataset)
            elif self.sampleMethod == 'undersample':
                return self.randomUnderSampling(trainingDataset)
            else:
                raise Exception('Wrong sample method parameter. Expected: float, "oversample", "undersample".')
        else:
            raise Exception('Wrong sample method parameter. Expected: float, "oversample", "undersample".')


    def bootstrapSampling(self, trainingDataset):
        numOfRowsInSample = int(trainingDataset.shape[0] * self.sampleMethod)
        return trainingDataset[numpy.random.choice(trainingDataset.shape[0], numOfRowsInSample, replace=True)]
    
    def randomOverSampling(self, trainingDataset):
        labels = trainingDataset[:, -1]
        classDistribution = collections.Counter(labels)
        majorityClass, majoritySamplesNum = max(classDistribution.items(), key=operator.itemgetter(1))

        # caluclate how many extra samples we have to take for each class
        numOfSamplesByClass = {key: majoritySamplesNum - value for key, value in classDistribution.items() if key != majorityClass}

        # initialize resulting indices with all rows from trainingDataset
        sampleIndices = range(trainingDataset.shape[0])

        for classVal, numOfSamples in numOfSamplesByClass.items():
            classIndicies = numpy.flatnonzero(labels == classVal)
            classIndex = numpy.random.randint(low=0, high=classDistribution[classVal], size=numOfSamples)

            sampleIndices = numpy.append(sampleIndices, classIndicies[classIndex])

        return trainingDataset[sampleIndices]

    def randomUnderSampling(self, trainingDataset):
        labels = trainingDataset[:, -1]
        classDistribution = collections.Counter(labels)
        minorityClass, minoritySamplesNum = min(classDistribution.items(), key=operator.itemgetter(1))

        # caluclate how many samples we have to take for each class
        numOfSamplesByClass = {key: minoritySamplesNum for key, value in classDistribution.items() if key != minorityClass}

        # start with empty indices
        sampleIndices = numpy.empty((0,), dtype=int)

        for classVal in classDistribution.keys():
            classIndices = numpy.flatnonzero(labels == classVal)

            if classVal in numOfSamplesByClass.keys():
                classIndex = numpy.random.choice(classIndices.shape[0], size=numOfSamplesByClass[classVal], replace=False)
            else:
                classIndex = slice(None)

            sampleIndices = numpy.append(sampleIndices, classIndices[classIndex])

        return trainingDataset[sampleIndices]







