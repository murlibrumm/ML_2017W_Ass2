#!/usr/bin/env python3

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn import preprocessing
import sys
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import warnings
import matplotlib.pyplot as plt
from collections import OrderedDict


class LastUpdatedOrderedDict(OrderedDict):
    "Store items in the order the keys were last added."
    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)


# GLOBALS
LOGGING          = False     # verbose logging output
SPLIT_DATA       = 0.2      # split dataset into training and testdata
DATAFRAME        = None     # our dataframe
MISSING_VALUES   = 'mean'   # how to deal with missing values (delete, mean, median, most_frequent)
SCALER           = None
CLASS_ENC        = preprocessing.LabelEncoder()

# PLOT EXPORT SETTINGS
EXPORT_PLOT      = True
X_LABEL          = 'Perceptrons per Layer (3 Layers)'
PLOT_FILE_NAME   = 'energy/neural-tanh-adam-adaptive-200.png'


# change the regressor values here!
# ALGORITHMS         = ['forest',       'knn',        'neural',    'bagging'] #algorithms to use ['forest', 'knn', 'bayes', 'neural']
# algorithmParameter = [(70, 70+1, 10), (1, 15+1, 1), (2, 20+1, 3), (1, 50+1, 5)] # set a parameter in range(start, end, jump)
ALGORITHMS         = ['neural']
algorithmParameter = [(2, 20+1, 3)]

# forest params (algorithmParameter controls n_estimators)
forestCriterion = 'mse' # "mse" mean squared error "mae" mean absolute error
forestMaxDepth  = None      # how deep can a tree be max; default: none

# knn params (algorithmParameter control n_neighbors)
knnWeights   = 'distance'   # weights: 1) 'uniform' (default): weighted equally. 2) 'distance': closer neighbors => more influence
knnAlgorithm = 'auto'      # algorithm to compute the NN: {'ball_tree', 'kd_tree', 'brute', 'auto}

# bagging regression params
# TODO

# neural MLP params (algorithmParameter controls hidden_layer_sizes, default: (100,))
neuralActivation = 'tanh' # (activation function for the hidden layer) : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
neuralSolver = 'adam' # (for the weight optimization): {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
neuralLearningRate = 'adaptive' # (Learning rate schedule for weight updates).: {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
neuralMaxIter = 200 # max_iter : int, optional, default 200

# filter warnings of the type UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    global DATAFRAME
    DATAFRAME = readDataset('datasets/energydata_complete.csv')
    printlog(DATAFRAME)
    DATAFRAME = encodeDataset(DATAFRAME)
    printlog("==========================================================================")
    printlog(DATAFRAME)
    # DATAFRAME = handleMissingValues(DATAFRAME)
    DATAFRAME = normalizeDataset(DATAFRAME)
    regressors = getRegressors()
    trainAndPredict(regressors)


def readDataset(path):
    # csv => DataFrame
    df = pd.read_csv(path)
    printlog('dataset size:' + str(df.shape))
    return df


def encodeDataset(df):
    return df.apply(lambda x: encodeLabels(x))


def trainAndPredict(regressors):
    resultsPerRegressor = LastUpdatedOrderedDict()

    # split into 80% training data, 20% test data
    train, test = train_test_split(DATAFRAME, test_size=SPLIT_DATA)

    # get training & test samples/targets
    training_samples, training_target = getSamplesAndTargets(train)
    test_samples,     actual_leagues  = getSamplesAndTargets(test)

    for (model, name) in regressors:
        # for each regressor, do the training and evaluation
        model.fit(training_samples, training_target)

        # predict the samples
        predicted_leagues = model.predict(test_samples)

        # perform cross validation
        X, y = getSamplesAndTargets(DATAFRAME)
        crossScoresMeanAE = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')
        crossScoresMeanSE = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
        
        try:
            crossScoresMeanSLE = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_log_error')
        except:
            print("Error at calculating neg_mean_squared_log_error, using mse instead")
            crossScoresMeanSLE = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')

        crossScoresMedianAE     = cross_val_score(model, X, y, cv=10, scoring='neg_median_absolute_error')
        crossScoresExplainedVar = cross_val_score(model, X, y, cv=10, scoring='explained_variance')
        crossScoresR2           = cross_val_score(model, X, y, cv=10, scoring='r2')

        # summarize the fit of the model
        crossScoresMean = (crossScoresMeanAE.mean(), crossScoresMeanSE.mean(), crossScoresMeanSLE.mean(), crossScoresMedianAE.mean(), crossScoresExplainedVar.mean(), crossScoresR2.mean())
        crossScoresStd  = (crossScoresMeanAE.std() * 2, crossScoresMeanSE.std() * 2, crossScoresMeanSLE.std() * 2, crossScoresMedianAE.std() * 2, crossScoresExplainedVar.std() * 2, crossScoresR2.std() * 2)
        printResults(crossScoresMean, crossScoresStd, actual_leagues, predicted_leagues, name)
        resultsPerRegressor[name] = crossScoresMean

    printRegressorReport(resultsPerRegressor)
    if EXPORT_PLOT:
        printPlot(resultsPerRegressor)


def encodeLabels(column):
    # any data columns except for the "date" column (which gets dropped anyway) are floating point numbers 
    if column.dtype == 'object' and column.name != 'date':
        column = column.apply(float) # transform all values in the column to floats
    else:
        return column


def getRegressors():
    # add the various regressors
    regressors = []
    for idx, val in enumerate(ALGORITHMS):
        for i in range(*algorithmParameter[idx]):
            if val == "forest":
                name = "Random Forests (n={0})".format(i)
                regressors.append(
                    (RandomForestRegressor(n_estimators=i, criterion=forestCriterion, max_depth=forestMaxDepth), name))
            if val == "knn":
                name = "kNN (n={0})".format(i)
                regressors.append(
                    (KNeighborsRegressor(n_neighbors=i, weights=knnWeights, algorithm=knnAlgorithm), name))
            if val == "bagging":
                name = "Bagging Regressor (estimators={0})".format(i)
                regressors.append(
                    (BaggingRegressor(n_estimators=i), name))
            if val == "neural":
                name = "Neural Network (layers={0})".format(i)
                regressors.append(
                    (MLPRegressor(hidden_layer_sizes=(i, i, i), activation=neuralActivation, solver=neuralSolver,
                                   learning_rate=neuralLearningRate, max_iter=neuralMaxIter), name))
    return regressors


# call this method first with the trainingdata
def normalizeDataset(data):
    # drop columns "date" and "lights"
    # (the latter would be a second thing to predict, but we wanna keep it simple... for now)
    # also, the random variables (r1, r2) should get dropped
    featureVal = data.drop(columns=['date', 'lights', 'rv1', 'rv2'])
    if 'Appliances' in data.columns:
        featureVal = featureVal.drop(columns=['Appliances'])

    SCALER = preprocessing.StandardScaler().fit(featureVal)
    scaled_DF = pd.DataFrame(SCALER.transform(featureVal))
    scaled_DF.columns = featureVal.columns
    scaled_DF.index = featureVal.index
    featureVal = scaled_DF

    norm_DF = pd.DataFrame(preprocessing.normalize(featureVal, norm='l2'))
    norm_DF.columns = featureVal.columns
    norm_DF.index = featureVal.index
    featureVal = norm_DF

    # re-add the Appliances column as it was before normalization of the rest
    if 'Appliances' in data.columns:
        featureVal = data.loc[:, 'Appliances':'Appliances'].join(featureVal)

    return featureVal


def getSamplesAndTargets(data):
    # get training samples (without Appliances)
    samples = data
    targets = None
    if 'Appliances' in data.columns:
        samples = data.drop(['Appliances'], axis=1)
        targets = data['Appliances'].values
    return samples, targets


def printResults(crossScoresMean, crossScoresStd, actual_leagues, predicted_leagues, regressor):
    print("\n", "=" * 80, "\n")
    print("=== Regressor:", regressor, "===\n")
    print("=== Cross Validation Results: ===\n",
          "Mean absolute error:    %0.2f (+/- %0.2f)\n" % (crossScoresMean[0], crossScoresStd[0]),
          "Mean squared error:     %0.2f (+/- %0.2f)\n" % (crossScoresMean[1], crossScoresStd[1]),
          "Mean squared log error: %0.2f (+/- %0.2f)\n" % (crossScoresMean[2], crossScoresStd[2]),
          "Median absolute error:  %0.2f (+/- %0.2f)\n" % (crossScoresMean[3], crossScoresStd[3]),
          "Explained Variance:     %0.2f (+/- %0.2f)\n" % (crossScoresMean[4], crossScoresStd[4]),
          "R2 Score:               %0.2f (+/- %0.2f)\n" % (crossScoresMean[5], crossScoresStd[5]))
    print()


def printRegressorReport(resultsPerRegressor):
    print()
    print("=" * 80)
    print("=== Report per Regressor: ===")
    printlog(resultsPerRegressor)
    resultFormat = '({:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f})'
    for name, results in resultsPerRegressor.items():
        print("=== %s ===" % name)
        printlog(results)
        print("(mae, mse, msle, median ae, explained variance, r2): ", resultFormat.format(*results))


# only plot if there is only one algorithm in the array!
def printPlot(resultsPerRegressor):
    MAE     = [results[0] for name, results in resultsPerRegressor.items()]
    MSE     = [results[1] for name, results in resultsPerRegressor.items()]
    MedAE   = [results[3] for name, results in resultsPerRegressor.items()]
    r2Score = [results[5] for name, results in resultsPerRegressor.items()]
    xAxis = list(range(*algorithmParameter[0]))
    printlog(list(range(*algorithmParameter[0])))
    printlog(MAE)
    printlog(MSE)
    printlog(MedAE)
    printlog(r2Score)

    fig = plt.figure(figsize=(8, 8))
    MAELine,     = plt.plot(xAxis, MAE,     label='Mean absolute error')
    #MSELine,     = plt.plot(xAxis, MSE,     label='Mean squared error')
    MedAELine,   = plt.plot(xAxis, MedAE,   label='Median absolute error')
    r2ScoreLine, = plt.plot(xAxis, r2Score, label='R2 Score')
    plt.legend(handles=[MAELine, MedAELine, r2ScoreLine])
    plt.ylabel('performance')
    plt.xlabel(X_LABEL)
    #plt.show()
    plt.savefig('figures/' + PLOT_FILE_NAME)
    plt.close(fig)


def printlog(message):
    if (LOGGING):
        print(message)


if __name__ == '__main__':
    exit = main()
    sys.exit(exit)
