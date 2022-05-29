import glob
import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from time import time
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
import pickle

sns.set(color_codes=True)

# Path to .CSV files: Read only
DATA_PATH = 'csv/'
# Path to your generated .csv files: Save your .csv files here
DELIVERABLE_PATH = 'out/'
# Path to models output: Save your models here
MODEL_PATH = 'models/'

anomaly_value = 100


def ageConverter(x):

    try:
        days = int(x.split(' ')[0])
        time = x.split(' ')[2]
        hours, minutes, seconds = time.split(':')
        total_seconds = int(seconds) + int(minutes)*60 + int(hours)*60*60 + days*24*60*60
    except:
        total_seconds = -anomaly_value
    return total_seconds


def datetoint(x):

    try:
        year = int(x.split('/')[2])
        month = int(x.split('/')[1])
        day = int(x.split('/')[0])
        dt_time = (datetime.date(year, month, day))
        purchase = 10000*dt_time.year + 100*dt_time.month + dt_time.day
    except:
        purchase = -anomaly_value
    return purchase


def process_data(files, oversample = False):

    objects = {'SACK' : 0, 'ALIEN': 1, 'SHOVEL' : 2, 'HAYSTACK' : 3, 'BUCKET' : 4}
    animals = {'COW' : 0, 'SHEEP' : 1, 'DOG' : 2, 'PIG' : 3, 'CHICKEN' : 4}

    data = []
    for file in files:
        data.append(pd.read_csv(file))

    animals_data = data[0]
    main_data = data[1]
    objects_data = data[2]

    # we dont need to know the photo resolution
    animals_data = animals_data.drop(['Resolution_Width', 'Resolution_Height'], axis=1)
    main_data = main_data.drop(['Resolution_Width', 'Resolution_Height'], axis=1)
    objects_data = objects_data.drop(['Resolution_Width', 'Resolution_Height'], axis=1)

    # join our dataframes based on the Date and Frame Columns
    joint_data = pd.merge(animals_data, main_data, on=['Date', 'Frame'])
    df = pd.merge(objects_data, joint_data, on = ['Date', 'Frame'])
    df = df.drop_duplicates()
    df = df.drop(['Date', 'Frame', 'Animal_Age', 'Temperature (F)', 'BB_X1_A', 'BB_Y1_A', 'BB_X2_A'], axis=1) # categories with very little correlation to UFO

    # we note that purchase date and object date have missing values, these correspond to entries with Alien listed
    #df['Temperature (F)'] = df['Temperature (F)'].apply(lambda x: int(x*100)) # convert temperature to int
    df['Object_Type'] = df['Object_Type'].apply(lambda x: objects[x]) # convert from object to int based on predefined dictionary
    df['Animal_Type'] = df['Animal_Type'].apply(lambda x: animals[x]) # convert from object to int based on predefined dictionary

    # convert object types to integers
    df['Object_Age'] = df['Object_Age'].apply(lambda x: ageConverter(x))
    df['Purchase_Date'] = df['Purchase_Date'].apply(lambda x: datetoint(x))

    print(df.isna().sum()/len(df)*100) # check for columns with missing values
    print(df.dtypes) # check we have suitable data types
    print(df['Is_UFO'].value_counts()) # we have an unbalanced dataset

    # standardisation of data
    #std_scaler = MinMaxScaler()
    #df_scaled = std_scaler.fit_transform(df.to_numpy())
    #df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

    correlation_df = df.corr()
    #plt.figure(figsize=(20,20))
    #sns_plot = sns.heatmap(correlation_df,cmap='BrBG',annot=True)
    #figure = sns_plot.get_figure()
    #figure.savefig('out\\Correlations_Heat_Map.png', dpi=400)
    df.to_csv('out\\Processed_Features.csv', index=False)
    correlation_df.to_csv('out\\Correlations.csv', index=True)

    y = df['Is_UFO']
    X = df.drop(['Is_UFO'], axis=1)
    #names = df.columns
    #for name in names:
    #    bplot = sns.boxplot(x=df[str(name)])
    #    figure = bplot.get_figure()
    #    figure_title = 'out\\' + name + '_boxplot'
    #    figure.savefig(figure_title)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42, shuffle=True)
    x_train, x_test, z_train, z_test = train_test_split(X_train,y_train, test_size=0.1, random_state=42, shuffle=True)

    # x_train, z_train - training
    # X_test, y_test - testing
    # x_test, z_test - validation

    if oversample:
        # synthetic dataset set augmentation, we only perform on training data
        print('Before Smote:', z_train.value_counts())
        sm = SMOTE()
        x_train, z_train = sm.fit_resample(x_train, z_train)
        print('After Smote:', z_train.value_counts())
    else:
        # undersample
        print("Before undersampling: ", Counter(y_train))
        undersample = RandomUnderSampler(sampling_strategy='majority')
        x_train, z_train = undersample.fit_resample(x_train, z_train)
        print("After undersampling: ", Counter(z_train))

    return x_train, z_train, x_test, z_test, X_test, y_test


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))
    return clf


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on roc_auc score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    probas = clf.predict_proba(features)
    end = time()

    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    scores = roc_auc_score(target.values, probas[:,1].T)
    return scores


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on roc_auc score. '''

    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, X_train.shape[0]))

    # Train the classifier
    clf = train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print ("ROC_AUC score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print ("ROC_AUC score for Validation set: {:.4f}.\n".format(predict_labels(clf, X_test, y_test)))

    file_name = 'models\\' + clf.__class__.__name__ + '_model.pkl'
    with open(file_name, "wb") as open_file:
        pickle.dump(clf, open_file)


def clf_test_roc_score(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    scores = roc_auc_score(y_test, probas[:,1].T)
    return scores


def clf_test_f1score(clf, X_train, y_train, X_test, y_test):
    dict_mapping = {0 : True, 1 : False}
    y_pred = []

    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)

    for proba in probas:
        position = np.argmax(proba)
        y_pred.append(dict_mapping[position])

    f1 = f1_score(y_test, y_pred, average=None)
    print(clf.__class__.__name__ + ' F1 SCORE', f1[0])
    return f1[0]


def clf_test_save_predictions(clf, X_train, y_train, X_test, y_test):
    dict_mapping = {0 : True, 1 : False}
    y_pred = []

    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)

    for proba in probas:
        position = np.argmax(proba)
        y_pred.append(dict_mapping[position])

    df_predictions = pd.DataFrame(X_test)
    df_predictions['Predictions'] = y_pred
    df_predictions['Actual'] = y_test

    df_predictions.to_csv('out\\Predictions.csv', index=True)


def clf_test_confusion_matrix(clf, X_train, y_train, X_test, y_test):

    dict_mapping = {0 : True, 1 : False}
    y_pred = []

    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)

    for proba in probas:
        position = np.argmax(proba)
        y_pred.append(dict_mapping[position])

    conf_matrix = (confusion_matrix(y_pred, y_test, normalize='pred'))

    df_cm = pd.DataFrame(conf_matrix, index = ['False', 'True'], columns = ['True', 'False'])
    #plt.figure(figsize = (10,7))
    #bplot = sns.heatmap(df_cm, annot=True)

    #figure = bplot.get_figure()
    #figure_title = 'out\\confusion_matrix\\' + clf.__class__.__name__ + '_Confusion Matrix'
    #figure.savefig(figure_title)
    return conf_matrix


def train_test_machinemodels(x_train, z_train, X_test, y_test):
    # Initialize the models using a random state were applicable.

    x_train = x_train.to_numpy()
    z_train = z_train
    X_test = X_test.to_numpy()
    y_test = y_test

    print(z_train)
    clf_list = [GaussianNB(),
                AdaBoostClassifier(random_state = 42),
                RandomForestClassifier(random_state = 42),
                LogisticRegression(random_state = 42),
                DecisionTreeClassifier(random_state = 42)]

    train_feature_list = [x_train]
    train_target_list = [z_train]


    # Execute the 'train_predict' function for each of the classifiers and each training set size
    for clf in clf_list:
        for a, b in zip(train_feature_list, train_target_list):
            train_predict(clf, a, b, X_test, y_test)


    ### Visualize all of the classifiers
    for clf in clf_list:
        x_graph = []
        y_graph = []
        for a, b in zip(train_feature_list, train_target_list):
            y_graph.append(clf_test_roc_score(clf, a, b, X_test, y_test))
            clf_test_confusion_matrix(clf, a, b, X_test, y_test)
            clf_test_f1score(clf, a, b, X_test, y_test)
            clf_test_save_predictions(clf, a, b, X_test, y_test)
            x_graph.append(len(a))
        plt.scatter(x_graph,y_graph)
        plt.plot(x_graph,y_graph, label = clf.__class__.__name__)

    plt.title('Comparison of Different Classifiers')
    plt.xlabel('Training Size')
    plt.ylabel('ROC_AUC score on Validation set')
    plt.legend()
    plt.savefig('out\\Classifier_Performance')


if __name__ == '__main__':
    # Load all files from the data path
    files = glob.glob(DATA_PATH + "*.csv")
    x_train, z_train, x_test, z_test, X_test, y_test = process_data(files, oversample=False)
    train_test_machinemodels(x_train, z_train, X_test, y_test)
