import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn import tree

#Bagging method
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor

#Boosting method
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

#Random Forest method
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
from sklearn import preprocessing
import matplotlib.pyplot as plt

import pandas as pd
import numpy



import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def open_file(file_name):
    data = pd.read_csv(file_name)
    return data

def write_file(data, fileName):
    data.to_csv(fileName)


def principal_components_analysis(data, n_components):
    # import data
    num_features = len(data.columns) - 1

    cols = data.columns
    num_cols = data._get_numeric_data().columns
    nominal_cols = list(set(cols) - set(num_cols))

    data[nominal_cols] = convert_data_to_numeric(data[nominal_cols])

    features = data[list(range(0, num_features))]
    target = data[[num_features]]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))

    # Model declaration
    if n_components < 1:
        pca = PCA(n_components = n_components, svd_solver = 'full')
    else:
        pca = PCA(n_components = n_components)

    # Model training
    pca.fit(features)

    # Model transformation
    new_feature_vector = pca.transform(features)

    # Model information:
    print('\nModel information:\n')
    print('Number of components elected: ' + str(pca.n_components))
    print('New feature dimension: ' + str(pca.n_components_))
    print('Variance of every feature: ' + str(sum(pca.explained_variance_ratio_ )))
    print('Variance of every feature: ' + str(pca.explained_variance_ratio_ ))

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

    # Print complete dictionary
    # print(pca.__dict__)

def convert_data_to_numeric(data):

    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        dict = numpy.unique(numpy_data[:,i])
        # print(dict)
        for j in range(len(dict)):
            #print(numpy.where(numpy_data[:,i] == dict[j]))
            temp[numpy.where(numpy_data[:,i] == dict[j])] = j

        numpy_data[:,i] = temp

    return numpy_data

def remove_columns(data, column):
    data = data.drop(column, axis=1, inplace=True)
    return data

def remove_outliers(data, feature, outlier_value):
    outliers = data.loc[data[feature] >= outlier_value, feature].index
    data.drop(outliers, inplace=True)
    return data

def replace_missing_values_with_mode(data, features):
    features = data[features]
    columns = features.columns
    mode = data[columns].mode()
    data[columns] = data[columns].fillna(mode.iloc[0])
    return data

def replace_missing_values_with_constant(data):
    #data['Alley'] = data['Alley'].fillna('NOACCESS')

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        data[col] = data[col].fillna('0')

    """data['FireplaceQu'] = data['FireplaceQu'].fillna('NoFP')

    for col in ('GarageType', 'GarageFinish', 'GarageQual'):
        data[col] = data[col].fillna('NoGRG')

    data['Fence'] = data['Fence'].fillna('NOFENCE')
    data['MiscFeature'] = data['MiscFeature'].fillna('NOMISC')
"""
    return data

def z_score_normalization(data):
    # import data
    num_features = len(data.columns) - 1

    cols = data.columns
    num_cols = data._get_numeric_data().columns
    nominal_cols = list(set(cols) - set(num_cols))

    data[nominal_cols] = convert_data_to_numeric(data[nominal_cols])

    features = data[list(range(0, num_features))]
    target = data[[num_features]]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))

    # Data standarization
    standardized_data = preprocessing.scale(features)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(standardized_data[:10])


def min_max_scaler(data):
    """# import data
    num_features = len(data.columns) - 1
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    nominal_cols = list(set(cols) - set(num_cols))
    data[nominal_cols] = convert_data_to_numeric(data[nominal_cols])
    features = data[list(range(1, num_features))]
    target = data[[num_features]]"""

    features = data[:,0:-1]
    target = data[:,-1]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))

    # Data normalization
    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.fit(features)

    # Model information:
    print('\nModel information:\n')
    print('Data min: ' + str(min_max_scaler.data_min_))
    print('Data max: ' + str(min_max_scaler.data_max_))

    new_feature_vector = min_max_scaler.transform(features)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

    new_data = np.append(new_feature_vector, target.reshape(target.shape[0], -1), axis=1)
    print('\nNew array\n')
    print(new_data)

    return new_data

def decision_tree(data):
    num_features = len(data.columns) - 1
    features = data[list(range(1, num_features))]
    target = data[[num_features]]
    print(features)
    print(target)
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(features,
                         target,
                         test_size=0.25)
    # Model declaration
    """
    Parameters to select:
    criterion: "mse"
    max_depth: maximum depth of tree, default: None
    """
    dec_tree_reg = DecisionTreeRegressor(criterion='mse', max_depth=5)
    dec_tree_reg.fit(data_features_train, data_targets_train)

    # Model evaluation
    test_data_predicted = dec_tree_reg.predict(data_features_test)

    error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)

    print('Total Error: ' + str(error))

def mlp_classifier(data):
    #load data
    num_features = len(data.columns) - 1
    features = data[list(range(1, num_features))]
    targets = data[[num_features]]
    print(features)
    print(targets)

    # Data splitting
    features_train, features_test, targets_train, targets_test = data_splitting(
        features,
        targets,
        0.25)

    # Model declaration
    """
    Parameters to select:
    hidden_layer_sizes: its an array in which each element represents a new layer with "n" neurones on it
            Ex. (3,4) = Neural network with 2 layers: 3 neurons in the first layer and 4 neurons in the second layer
            Ex. (25) = Neural network with one layer and 25 neurons
            Default = Neural network with one layer and 100 neurons
    activation: "identity", "logistic", "tanh" or "relu". Default: "relu"
    solver: "lbfgs", "sgd" or "adam" Default: "adam"
    ###Only used with "sgd":###
    learning_rate_init: Neural network learning rate. Default: 0.001
    learning_rate: Way in which learning rate value change through iterations.
            Values: "constant", "invscaling" or "adaptive"
    momentum: Default: 0.9
    early_stopping: The algorithm automatic stop when the validation score is not improving.
            Values: "True" or "False". Default: False
    """
    neural_net = MLPClassifier(
        hidden_layer_sizes=(50),
        activation="relu",
        solver="adam"
    )
    neural_net.fit(features_train, targets_train.values.ravel())
        # Model evaluation
    test_data_predicted = neural_net.predict(features_test)
    score = metrics.accuracy_score(targets_test, test_data_predicted)
    logger.debug("Model Score: %s", score)
def data_splitting(data_features, data_targets, test_size):
    """
    This function returns four subsets that represents training and test data
    :param data: numpy array
    :return: four subsets that represents data train and data test
    """
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(data_features,
                         data_targets,
                         test_size = test_size)

    return data_features_train, data_features_test, data_targets_train, data_targets_test


def decision_tree(data):
    num_features = len(data.columns) - 1

    features = data[list(range(1, num_features))]
    target = data[[num_features]]

    print(features)
    print(target)
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(features,
                         target,
                         test_size=0.25)

    # Model declaration
    """
    Parameters to select:
    criterion: "mse"
    max_depth: maximum depth of tree, default: None
    """
    dec_tree_reg = DecisionTreeRegressor(criterion='mse', max_depth=5)
    dec_tree_reg.fit(data_features_train, data_targets_train)

    # Model evaluation
    test_data_predicted = dec_tree_reg.predict(data_features_test)
    error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)
    print('Total Error: ' + str(error))

def mlp_classifier(data):
    #load data
    num_features = len(data.columns) - 1

    features = data[list(range(1, num_features))]
    targets = data[[num_features]]

    print(features)
    print(targets)

    # Data splitting
    features_train, features_test, targets_train, targets_test = data_splitting(
        features,
        targets,
        0.25)

    # Model declaration
    """
    Parameters to select:
    hidden_layer_sizes: its an array in which each element represents a new layer with "n" neurones on it
            Ex. (3,4) = Neural network with 2 layers: 3 neurons in the first layer and 4 neurons in the second layer
            Ex. (25) = Neural network with one layer and 25 neurons
            Default = Neural network with one layer and 100 neurons
    activation: "identity", "logistic", "tanh" or "relu". Default: "relu"
    solver: "lbfgs", "sgd" or "adam" Default: "adam"
    ###Only used with "sgd":###
    learning_rate_init: Neural network learning rate. Default: 0.001
    learning_rate: Way in which learning rate value change through iterations.
            Values: "constant", "invscaling" or "adaptive"
    momentum: Default: 0.9
    early_stopping: The algorithm automatic stop when the validation score is not improving.
            Values: "True" or "False". Default: False
    """
    neural_net = MLPClassifier(
        hidden_layer_sizes=(50),
        activation="relu",
        solver="adam"
    )
    neural_net.fit(features_train, targets_train.values.ravel())
    # Model evaluation
    test_data_predicted = neural_net.predict(features_test)
    score = metrics.accuracy_score(targets_test, test_data_predicted)

    logger.debug("Model Score: %s", score)

def ensemble_methods_classifiers(data):
            #load data
    num_features = len(data.columns) - 1

    features = data[list(range(1, num_features))]
    targets = data[[num_features]]

    print(features)
    print(targets)
    # Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = data_splitting(
        features,
        targets,
        0.25)

    # Model declaration
    """
    Parameters to select:
    n_estimators: The number of base estimators in the ensemble.
            Values: Random Forest and Bagging. Default 10
                    AdaBoost. Default: 50
    ###Only for Bagging and Boosting:###
    base_estimator: Base algorithm of the ensemble. Default: DecisionTree
    ###Only for Random Forest:###
    criterion: "entropy" or "gini": default: gini
    max_depth: maximum depth of tree, default: None
    """
    names = ["Bagging Classifier", "AdaBoost Classifier", "Random Forest Classifier"]
    models = [
        BaggingClassifier(
            base_estimator=tree.DecisionTreeClassifier(
                criterion='gini',
                max_depth=10)
        ),
        AdaBoostClassifier(
            n_estimators=10,
            base_estimator=tree.DecisionTreeClassifier(
                criterion='gini',
                max_depth=10)
        ),
        RandomForestClassifier(
            criterion='gini',
            max_depth=10
        )
    ]

    for name, em_clf in zip(names, models):
        logger.info("###################---" + name + "---###################")
        em_clf.fit(data_features_train, data_targets_train.values.ravel())

        # Model evaluation
        test_data_predicted = em_clf.predict(data_features_test)
        score = metrics.accuracy_score(data_targets_test, test_data_predicted)

        logger.debug("Model Score: %s", score)

def xgboost(data):
    for f in data.columns:
        if data[f].dtype=='object':

            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))

    train_y = data.price_doc.values
    train_X = data.drop(["id", "timestamp", "price_doc"], axis=1)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)


    # plot the important features #
    fig, ax = plt.subplots(figsize=(12,18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()

def data_splitting(data_features, data_targets, test_size):
    """
    This function returns four subsets that represents training and test data
            :param data: numpy array
            :return: four subsets that represents data train and data test
            """
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(data_features,
                         data_targets,
                         test_size = test_size)

    return data_features_train, data_features_test, data_targets_train, data_targets_test

if __name__ == '__main__':
    print("------------DATA SIN LIMPIEZA------------")
    #data = open_file('train.csv')
    data = open_file('test.csv')
    print(data)
    remove_outliers(data, 'full_sq', 5000)
    remove_outliers(data, 'max_floor', 80)
    remove_outliers(data, 'num_room', 15)

    print("------------LIMPIEZA------------")


    remove_columns(data, ['max_floor','timestamp','hospital_beds_raion','state', 'num_room'])
    replace_missing_values_with_mode(data, ['build_year','material','kitch_sq','floor'])
    data = data.fillna(0)
    convert_data_to_numeric(data)
    #min_max_scaler(data)

    principal_components_analysis(data, 72)
    z_score_normalization(data)

    print(data)

    print("---------------------------------------Arbol de Decision---------------------------------------------")

    new_data= data
    new_data = convert_data_to_numeric(data)


    feature_vector = new_data[:,0:-1]
    targets = new_data[:,-1]

    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(feature_vector,
                         targets,
                         test_size=0.25, random_state=24)

    # Model declaration
    """
    Parameters to select:
    criterion: "mse"
    max_depth: maximum depth of tree, default: None
    """
    dec_tree_reg = DecisionTreeRegressor(criterion='mse', max_depth=40)
    dec_tree_reg.fit(data_features_train, data_targets_train)

    # Model evaluation
    test_data_predicted = dec_tree_reg.predict(data_features_test)

    error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)

    print('Total Error: ' + str(error))

    #decision_tree(data)
    mlp_classifier(data)

    decision_tree(data)
    mlp_classifier(data)
    ensemble_methods_classifiers(data)
    #xgboost(data)
    write_file(data, 'outputtestfinal.csv')
