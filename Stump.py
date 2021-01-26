import random
import collections
import operator
import numpy

class Stump:
    """
     dict mapping possible values to classes
     e.g for safety
     {
        "low" : "class1"
        "med" : "class2"
        "high" : "class1"
     }
    """
    #decisions

    """
    trainingDataset = (numpy2darray)
    """
    def __init__(self, trainingDataset, numOfFeatures, possibleClasses):
        self.possibleClasses = possibleClasses
        self.attrIndex, self.decisions = self.train(trainingDataset, numOfFeatures)

    """
    count number of classes with given attribue
    returns:
    groups = 
     {
        "low" :  ( 10 , 1, ...),
        "med" : ...,
        "high" : ..., 
     }
    """
    def count(self, trainingDataset, attrIndex):
       
        attrColumn = trainingDataset[:, attrIndex]
        groups = {}
        for attrVal in self.possibleAttrValues(attrColumn):
            attrFilter = attrColumn == attrVal
            groups[attrVal] = collections.Counter(trainingDataset[attrFilter][:, -1])

        return groups

    def possibleAttrValues(self, attrColumn):
        return numpy.unique(attrColumn)


    def calculateDecisions(self, groups):
        decisions = {}
        for attrVal, group in groups.items():
            decisions[attrVal] =  max(group.items(), key=operator.itemgetter(1))[0]

        return decisions


    def train(self, trainingDataset, numOfFeatures):
        bestScore = float('inf')
        possibleAttrIndexes = list(range(trainingDataset.shape[1] - 1))

        features = random.sample(possibleAttrIndexes, numOfFeatures)

        for attrIndex in features:
            groups = self.count(trainingDataset, attrIndex)
            giniScore = self.gini(groups, trainingDataset.shape[0])

            if giniScore < bestScore:
                bestScore = giniScore
                bestIndex = attrIndex
                bestGroups = groups

        bestDecisions = self.calculateDecisions(bestGroups)

        return bestIndex, bestDecisions

    def gini(self, groups, numOfInstances):
        gini = 0.0
        for group in groups.values():
            size = float(sum(group.values()))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for count in group.values():
                p = count / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / numOfInstances)
        return gini

    def __str__(self):
        return f'Stump: {self.attrIndex}: \t{self.decisions}'

    # split dateset into groups based on decisions
    """
    returns one of possible classes
    """
    def decide(self, row):
        attrValue = row[self.attrIndex]
        return self.decisions.get(attrValue)

    
    


