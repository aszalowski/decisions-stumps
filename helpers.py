import pandas as pd
import csv
import numpy
from StumpForest import StumpForest

from sklearn.model_selection import KFold 
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import json 

def loadCSV(filename):
    df = pd.read_csv(filename, header=None)
    return df.values

def crossValidationSplitter(numOfFolds):
    return KFold(numOfFolds, random_state=None, shuffle=True)

def evaluateParameters(dataset):
    splitter = crossValidationSplitter(4)

    numOfStumps = [10, 100, 500, 1000] 
    numOfFeatures = [1, 2, 4, 6] 
    
    scores = {
            'forest_bootstrap' : {'stumps' : {n: 0 for n in numOfStumps}, 'features' : {n: 0 for n in numOfFeatures}},
            'forest_undersampling' : {'stumps' : {n: 0 for n in numOfStumps}, 'features' : {n: 0 for n in numOfFeatures}},
            'forest_oversampling' : {'stumps' : {n: 0 for n in numOfStumps}, 'features' : {n: 0 for n in numOfFeatures}},
    }

    for trainIdx, testIdx in splitter.split(dataset):

        trainingDataset = dataset[trainIdx]
        testDataset = dataset[testIdx]

        for n in numOfStumps:
            forest_bootstrap = StumpForest(trainingDataset, numOfStumps=n, sampleMethod=0.5)
            forest_undersampling = StumpForest(trainingDataset, numOfStumps=n, sampleMethod='undersample')
            forest_oversampling = StumpForest(trainingDataset, numOfStumps=n, sampleMethod='oversample')

            predictions = [forest_bootstrap.predict(row) for row in testDataset]
            scores['forest_bootstrap']['stumps'][n] += metrics.accuracy_score(testDataset[:, -1], predictions)

            predictions = [forest_undersampling.predict(row) for row in testDataset]
            scores['forest_undersampling']['stumps'][n] += metrics.accuracy_score(testDataset[:, -1], predictions)

            predictions = [forest_oversampling.predict(row) for row in testDataset]
            scores['forest_oversampling']['stumps'][n] += metrics.accuracy_score(testDataset[:, -1], predictions)

        for n in numOfFeatures:
            forest_bootstrap = StumpForest(trainingDataset, numOfStumps=n, sampleMethod=0.5)
            forest_undersampling = StumpForest(trainingDataset, numOfStumps=n, sampleMethod='undersample')
            forest_oversampling = StumpForest(trainingDataset, numOfStumps=n, sampleMethod='oversample')

            predictions = [forest_bootstrap.predict(row) for row in testDataset]
            scores['forest_bootstrap']['features'][n] += metrics.accuracy_score(testDataset[:, -1], predictions)

            predictions = [forest_undersampling.predict(row) for row in testDataset]
            scores['forest_undersampling']['features'][n] += metrics.accuracy_score(testDataset[:, -1], predictions)

            predictions = [forest_oversampling.predict(row) for row in testDataset]
            scores['forest_oversampling']['features'][n] += metrics.accuracy_score(testDataset[:, -1], predictions)

    mean_scores = {}
    for key, scores_dict in scores.items():
        mean_scores[key] = {score_name: {key: value / 4.0 for key, value in score_dict.items()} for score_name, score_dict in scores_dict.items()}

    with open('parameters_tests.json', 'w') as f:
        json.dump(mean_scores, f, indent=4)


def compareWithExternal(dataset):
    splitter = crossValidationSplitter(4)
    
    scores = {
            'forest_bootstrap' : {'acc' : [], 'prec' : [], 'rec' : []},
            'forest_undersampling' : {'acc' : [], 'prec' : [], 'rec' : []},
            'forest_oversampling' : {'acc' : [], 'prec' : [], 'rec' : []},
            'sklearn_bootstrap' : {'acc' : [], 'prec' : [], 'rec' : []},
            'sklearn_balanced' : {'acc' : [], 'prec' : [], 'rec' : []},
    } 

    for trainIdx, testIdx in splitter.split(dataset):
        trainingDataset = dataset[trainIdx]
        testDataset = dataset[testIdx]

        forest_bootstrap = StumpForest(trainingDataset, numOfStumps=1000, sampleMethod=0.5)
        forest_undersampling = StumpForest(trainingDataset, numOfStumps=1000, sampleMethod='undersample')
        forest_oversampling = StumpForest(trainingDataset, numOfStumps=1000, sampleMethod='oversample')

        sklearn_bootstrap = RandomForestClassifier(n_estimators=1000, max_depth=1, bootstrap=True)
        sklearn_balanced = RandomForestClassifier(n_estimators=1000, max_depth=1, class_weight='balanced')

        X, y = trainingDataset[:, : -1], trainingDataset[:, -1]

        X = numpy.apply_along_axis(LabelEncoder().fit_transform, 0, X)
        y = LabelEncoder().fit_transform(y)
        X_test = numpy.apply_along_axis(LabelEncoder().fit_transform, 0, testDataset[:, : -1])
        y_test = LabelEncoder().fit_transform(testDataset[:, -1])

        sklearn_bootstrap.fit(X, y)
        sklearn_balanced.fit(X, y)


        predictions = [forest_bootstrap.predict(row) for row in testDataset]
        scores['forest_bootstrap']['acc'].append(metrics.accuracy_score(testDataset[:, -1], predictions))
        scores['forest_bootstrap']['prec'].append(metrics.precision_score(testDataset[:, -1], predictions, average='weighted', zero_division=0))
        scores['forest_bootstrap']['rec'].append(metrics.recall_score(testDataset[:, -1], predictions, average='weighted', zero_division=0))

        predictions = [forest_undersampling.predict(row) for row in testDataset]
        scores['forest_undersampling']['acc'].append(metrics.accuracy_score(testDataset[:, -1], predictions))
        scores['forest_undersampling']['prec'].append(metrics.precision_score(testDataset[:, -1], predictions, average='weighted', zero_division=0))
        scores['forest_undersampling']['rec'].append(metrics.recall_score(testDataset[:, -1], predictions, average='weighted', zero_division=0))

        predictions = [forest_oversampling.predict(row) for row in testDataset]
        scores['forest_oversampling']['acc'].append(metrics.accuracy_score(testDataset[:, -1], predictions))
        scores['forest_oversampling']['prec'].append(metrics.precision_score(testDataset[:, -1], predictions, average='weighted', zero_division=0))
        scores['forest_oversampling']['rec'].append(metrics.recall_score(testDataset[:, -1], predictions, average='weighted', zero_division=0))

        predictions = sklearn_bootstrap.predict(X_test)
        scores['sklearn_bootstrap']['acc'].append(metrics.accuracy_score(y_test, predictions))
        scores['sklearn_bootstrap']['prec'].append(metrics.precision_score(y_test, predictions, average='weighted', zero_division=0))
        scores['sklearn_bootstrap']['rec'].append(metrics.recall_score(y_test, predictions, average='weighted', zero_division=0))

        predictions = sklearn_balanced.predict(X_test)
        scores['sklearn_balanced']['acc'].append(metrics.accuracy_score(y_test, predictions))
        scores['sklearn_balanced']['prec'].append(metrics.precision_score(y_test, predictions, average='weighted', zero_division=0))
        scores['sklearn_balanced']['rec'].append(metrics.recall_score(y_test, predictions, average='weighted', zero_division=0))

    mean_scores = {}
    for key, scores_dict in scores.items():
        mean_scores[key] = {score_name: sum(score_values) / len(score_values) for score_name, score_values in scores_dict.items()}

    with open('sklearn_comparison.json', 'w') as f:
        json.dump(mean_scores, f, indent=4)



