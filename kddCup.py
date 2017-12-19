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
LOGGING          = True     # verbose logging output
SPLIT_DATA       = 0.2      # split dataset into training and testdata
DATAFRAME        = None     # our dataframe
MISSING_VALUES   = 'mean'   # how to deal with missing values (delete, mean, median, most_frequent)
SCALER           = None
CLASS_ENC        = preprocessing.LabelEncoder()

# PLOT EXPORT SETTINGS
EXPORT_PLOT      = True
X_LABEL          = 'Neighbors'
PLOT_FILE_NAME   = 'kdd_knn.png'


# change the regressor values here!
#ALGORITHMS         = ['forest',       'knn',        'neural',    'bagging'] #algorithms to use ['forest', 'knn', 'bayes', 'neural']
#algorithmParameter = [(70, 70+1, 10), (1, 15+1, 1), (2, 20+1, 3), (1, 50+1, 5)] # set a parameter in range(start, end, jump)
ALGORITHMS         = ['knn']
algorithmParameter = [(10, 100+1, 5)]

# forest params (algorithmParameter controls n_estimators)
forestCriterion = 'mse' # "mse" mean squared error "mae" mean absolute error
forestMaxDepth  = None      # how deep can a tree be max; default: none

# knn params (algorithmParameter control n_neighbors)
knnWeights   = 'distance'   # weights: 1) 'uniform' (default): weighted equally. 2) 'distance': closer neighbors => more influence
knnAlgorithm = 'auto'      # algorithm to compute the NN: {'ball_tree', 'kd_tree', 'brute', 'auto}

# bagging regression params
# TODO

# neural MLP params (algorithmParameter controls hidden_layer_sizes, default: (100,))
neuralActivation = 'relu' # (activation function for the hidden layer) : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
neuralSolver = 'lbfgs' # (for the weight optimization): {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
neuralLearningRate = 'invscaling'# (Learning rate schedule for weight updates).: {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
neuralMaxIter = 200 # max_iter : int, optional, default 200

# EXPORT PREDICTION
EXPORT_PREDICTION = False
EXPORT_MODEL      = None
EXPORT_FILE_NAME  = 'kdd_knn_60_prediction.csv'


# filter warnings of the type UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
#warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    global DATAFRAME
    DATAFRAME = readDataset('datasets/cup98ID.shuf.5000.train.csv')
    print(DATAFRAME)
    DATAFRAME = encodeDataset(DATAFRAME)
    print("==========================================================================")
    print(DATAFRAME)
    DATAFRAME = handleMissingValues(DATAFRAME)
    DATAFRAME = normalizeDataset(DATAFRAME)
    regressors = getRegressors()
    trainAndPredict(regressors)
    if EXPORT_PREDICTION:
        predictTestData()


def predictTestData():
    testDf = readDataset('datasets/cup98ID.shuf.5000.test.csv')
    encodedDf = encodeDataset(testDf)
    missingValuesDf = handleMissingValues(encodedDf)
    normalizedDf = normalizeDataset(missingValuesDf)

    # get training & test samples/targets
    test_samples,     actual_class = getSamplesAndTargets(normalizedDf)

    # prediction of the provided test data
    y_predicted_test = EXPORT_MODEL.predict(test_samples)
    testdata_id = testDf['CONTROLN']
    prediction = np.c_[np.array(testdata_id, dtype=object), y_predicted_test]
    np.savetxt('predictions/' + EXPORT_FILE_NAME, prediction, delimiter=",", fmt='%s', header='"CONTROLN","TARGET_D"', comments='')


def readDataset(path):
    # csv => DataFrame
    df = pd.read_csv(path)
    printlog('dataset size:' + str(df.shape))
    return df


def encodeDataset(df):
    return df.apply(lambda x: encodeLabels(x))


def handleMissingValues(dataframe):
    # drop all columns, where every value is null
    dataframe = dataframe.dropna(axis=1, how='all')
    if (MISSING_VALUES == 'delete'):
        # filter out missing values
        # https://stackoverflow.com/questions/27428954/drop-row-if-any-column-value-does-not-a-obey-a-condition-in-pandas
        dataframe = dataframe[~(dataframe == ' ').any(1)]

    if (MISSING_VALUES == 'median' or MISSING_VALUES == 'mean' or MISSING_VALUES == 'most_frequent'):
        # deal with missing values => mean
        dataframe.replace({' ': np.nan}, inplace=True)
        #dataframe.replace({'NaN': np.nan}, inplace=True)
        fill_NaN = preprocessing.Imputer(missing_values='NaN', strategy=MISSING_VALUES, axis=0)
        imputed_DF = pd.DataFrame(fill_NaN.fit_transform(dataframe))
        imputed_DF.columns = dataframe.columns
        imputed_DF.index = dataframe.index

        dataframe = imputed_DF
        printlog('dataset size after handling missing values:' + str(dataframe.shape))
    return dataframe


def trainAndPredict(regressors):
    global EXPORT_MODEL
    resultsPerRegressor = LastUpdatedOrderedDict()

    # split into 80% training data, 20% test data
    train, test = train_test_split(DATAFRAME, test_size=SPLIT_DATA)

    # get training & test samples/targets
    training_samples, training_target = getSamplesAndTargets(train)
    test_samples,     actual_leagues  = getSamplesAndTargets(test)

    for (model, name) in regressors:
        # for each regressor, do the training and evaluation
        model.fit(training_samples, training_target)
        EXPORT_MODEL = model

        # predict the samples
        predicted_leagues = model.predict(test_samples)

        # perform cross validation
        X, y = getSamplesAndTargets(DATAFRAME)
        crossScoresMeanAE       = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
        crossScoresMeanSE       = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        crossScoresMeanSLE      = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_mean_squared_log_error')
        crossScoresMedianAE     = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_median_absolute_error')
        crossScoresExplainedVar = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='explained_variance')
        crossScoresR2           = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='r2')

        # summarize the fit of the model
        crossScoresMean = (crossScoresMeanAE.mean(), crossScoresMeanSE.mean(), crossScoresMeanSLE.mean(), crossScoresMedianAE.mean(), crossScoresExplainedVar.mean(), crossScoresR2.mean())
        crossScoresStd  = (crossScoresMeanAE.std() * 2, crossScoresMeanSE.std() * 2, crossScoresMeanSLE.std() * 2, crossScoresMedianAE.std() * 2, crossScoresExplainedVar.std() * 2, crossScoresR2.std() * 2)
        printResults(crossScoresMean, crossScoresStd, actual_leagues, predicted_leagues, name)
        resultsPerRegressor[name] = crossScoresMean

    printRegressorReport(resultsPerRegressor)
    if EXPORT_PLOT:
        printPlot(resultsPerRegressor)


def encodeLabels(column):
    if column.dtype == 'object':
        containsIntValues = False
        containsStringValues = False
        for value in column.unique():
            maybeInt, isInt = intTryParse(value)
            if isInt:
                containsIntValues = True
            elif value != ' ' and value != '' and value != np.nan:
                containsStringValues = True

        if containsIntValues and not containsStringValues:
            return column
        else: # labelencoding!
            if containsIntValues:
                column = column.apply(str) # transform all integer values in this column to string
            column = column.fillna(' ')
            CLASS_ENC.fit(column)
            return CLASS_ENC.transform(column)
    else:
        return column


def intTryParse(value):
    try:
        return int(value), True
    except ValueError:
        return value, False


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
                name = "Neural Network (perceptrons per layer={0})".format(i)
                regressors.append(
                    (MLPRegressor(hidden_layer_sizes=(i, i, i), activation=neuralActivation, solver=neuralSolver,
                                   learning_rate=neuralLearningRate, max_iter=neuralMaxIter), name))
    return regressors


# call this method first with the trainingdata
def normalizeDataset(data):
    # drop id and TARGET_D
    featureVal = data.drop(columns=['CONTROLN'])
    if 'TARGET_D' in data.columns:
        featureVal = featureVal.drop(columns=['TARGET_D'])

    SCALER = preprocessing.StandardScaler().fit(featureVal)
    scaled_DF = pd.DataFrame(SCALER.transform(featureVal))
    scaled_DF.columns = featureVal.columns
    scaled_DF.index = featureVal.index
    featureVal = scaled_DF

    norm_DF = pd.DataFrame(preprocessing.normalize(featureVal, norm='l2'))
    norm_DF.columns = featureVal.columns
    norm_DF.index = featureVal.index
    featureVal = norm_DF

    if 'TARGET_D' in data.columns:
        featureVal = data.loc[:, 'TARGET_D':'TARGET_D'].join(featureVal)

    return featureVal


def getSamplesAndTargets(data):
    # get training samples (without TARGET_D)
    samples = data
    targets = None
    if 'TARGET_D' in data.columns:
        samples = data.drop(['TARGET_D'], axis=1)
        targets = data['TARGET_D'].values
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
