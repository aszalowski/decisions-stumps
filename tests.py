from StumpForest import StumpForest
import helpers
import numpy
import collections

# Comparions between our and external algos
dataset = helpers.loadCSV('car-data.csv')

helpers.evaluateParameters(dataset)
helpers.compareWithExternal(dataset)
