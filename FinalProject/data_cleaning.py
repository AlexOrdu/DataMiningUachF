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


if __name__ == '__main__':
    print("------------DATA SIN LIMPIEZA------------")
    data = open_file('train.csv')
    #data = open_file('test.csv')
    print(data)
    remove_outliers(data, 'full_sq', 5000)
    remove_outliers(data, 'max_floor', 80)
    remove_outliers(data, 'num_room', 15)
    remove_outliers(data, 'kitch_sq', 1750)
    print("------------LIMPIEZA------------")


    remove_columns(data, ['max_floor','timestamp','hospital_beds_raion','state', 'num_room', 'children_preschool'])
    replace_missing_values_with_mode(data, ['build_year','material','kitch_sq','floor'])
    data = data.fillna(0)
    convert_data_to_numeric(data)
    #min_max_scaler(data)

    principal_components_analysis(data, 75)
    z_score_normalization(data)

    print(data)

    print("---------------------------------------Arbol de Decision---------------------------------------------")

    #new_data= data
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
    dec_tree_reg = DecisionTreeRegressor(criterion='mse', max_depth=50)
    dec_tree_reg.fit(data_features_train, data_targets_train)

    # Model evaluation
    test_data_predicted = dec_tree_reg.predict(data_features_test)

    error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)

    print('Total Error: ' + str(error))
    print(data)
    #write_file(data, 'output.csv')

    #write_file(data, 'outputtrain.csv')
    write_file(data, '2itera/output.csv')
