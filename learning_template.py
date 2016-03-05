# XGB with function definitions for easy modification 

import numpy as np 
import pandas as pd 
import matplotlib.pylab as plt
import xgboost as xgb
import csv
import sklearn as sk 
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing
from timeit import default_timer as timer
from sys import platform as platform
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from scipy.stats.stats import pearsonr
# from sklearn import linear_model
# from sklearn.ensemble import GradientBoostingClassifier as GBC
# from sklearn.ensemble import ExtraTreesClassifier as EXT

#############################################################################
# Seed, prediction file name, and model parameters 
#############################################################################

test_prediction_file_string = "script_check_xgb_tree.csv"

#############################################################################

if platform == "linux2": 
    training_data_path = '~/Desktop/kaggle_homesite/train.csv'
    testing_data_path = '~/Desktop/kaggle_homesite/test.csv'
    sample_sub_path = '~/Desktop/kaggle_homesite/sample_submission.csv'
elif platform == "darwin":
    training_data_path = '~/Desktop/kaggle_competitions/Homesite/train.csv'
    testing_data_path = '~/Desktop/kaggle_competitions/Homesite/test.csv'
    sample_sub_path = '~/Desktop/kaggle_competitions/Homesite/sample_submission.csv'
else: 
    print "Couldn't recognize operating system"

# Read the files into pandas 
print "Reading in data..."
start = timer()
df = pd.read_csv(training_data_path, header = 0)
dftest = pd.read_csv(testing_data_path, header = 0)
print "Time to read in data:", timer() - start, "seconds"

#############################################################################
# Data Preprocessing 
#############################################################################

# convert date columns into year, month, and weekday number columns, convert labels into number labels for numpy

def clean_data(df): 
    # same cleaning for training and testing data
    df = df.drop('QuoteNumber', axis=1) # dropping quote number 
    
    # turn dates into numerical feature columns
    
    df['Date'] = pd.to_datetime(pd.Series(df['Original_Quote_Date']))
    df = df.drop('Original_Quote_Date', axis=1)
    
    df['Year'] = df['Date'].apply(lambda x: int(str(x)[:4]))
    df['Month'] = df['Date'].apply(lambda x: int(str(x)[5:7]))
    df['weekday'] = df['Date'].dt.dayofweek
    
    df = df.drop('Date', axis=1)
    
    # Fill in the empty values in the array with -999 
    # This data set came with -1 in the missing value locations
      
    df = df.fillna(value = -999)

    # Use a loop to convert all string/object labels into integer labels
    
    for f in df.columns:
        if df[f].dtype=='object':
            # print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values) + list(dftest[f].values))
            df[f] = lbl.transform(list(df[f].values))
            # dftest[f] = lbl.transform(list(dftest[f].values))
    return df

def drop_uniform_columns(df): 
    # drops columns that have same value for every entry - for both training and testing data 
    # these ended up being PropertyField6 and GeographicField10A 

    for col in list(df.columns.values): 
        if df[col].max() == df[col].min():
            print "Dropping ", col
            df = df.drop(col, axis=1) # dropping quote number 
    return df

def add_new_features(df): 
    # add new features to the data - use suggestions from kaggle forums
    start = timer()
    # count how many NA (i.e., -1) values are in each row, as well as how many NA values
    # (i.e. -1 values) are in in sub-category
    em2s = df.columns.get_loc("CoverageField1A")
    em2e = df.columns.get_loc("SalesField1A")

    em3s = df.columns.get_loc("SalesField1A")
    em3e = df.columns.get_loc("PersonalField1")

    em4s = df.columns.get_loc("PersonalField1")
    em4e = df.columns.get_loc("PropertyField1A")

    em5s = df.columns.get_loc("PropertyField1A")
    em5e = df.columns.get_loc("GeographicField1A")

    em6s = df.columns.get_loc("GeographicField1A")
    em6e = df.columns.get_loc("GeographicField64") + 1  
    
    df["empty2"] = df.ix[:,em2s:em2e].apply(lambda s: (s == -1).sum(), axis=1)
    df["empty3"] = df.ix[:,em3s:em3e].apply(lambda s: (s == -1).sum(), axis=1)
    df["empty4"] = df.ix[:,em4s:em4e].apply(lambda s: (s == -1).sum(), axis=1)
    df["empty5"] = df.ix[:,em5s:em5e].apply(lambda s: (s == -1).sum(), axis=1)
    df["empty6"] = df.ix[:,em6s:em6e].apply(lambda s: (s == -1).sum(), axis=1)    
    
    df["total_empty_count"]= df.apply(lambda s: (s == -1).sum(), axis=1)

    print "Time to add new features:", (timer() - start)/60., " minutes"
    
    return df



def pearson_correlation(x,y): 
    # calulate pearson correlation coefficient between two predictions x and y
    return pearsonr(x,y)



def grid_search(train, test, features): 
    # perform a standard grid search of parameters with 5-fold cross validation for each combo
    xgb_model = xgb.XGBClassifier()

    # brute force scan for all parameters, here are the tricks
    # usually max_depth is 6,7,8
    # learning rate is around 0.05, but small changes may make big diff
    # tuning min_child_weight subsample colsample_bytree can have 
    # much fun of fighting against overfit 
    # n_estimators is how many round of boosting
    # finally, ensemble xgboost with multiple seeds may reduce variance

    parameters = {
              'objective':['binary:logistic'],
              'learning_rate': [0.023], #so called `eta` value
              'max_depth': [8],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.83],
              'colsample_bytree': [0.77],
              'n_estimators': [1200], #number of trees, change it to 1000 for better results
              'seed': [1337]}

    clf = GridSearchCV(xgb_model, parameters, n_jobs=1, 
                   cv=StratifiedKFold(train['QuoteConversion_Flag'], n_folds=3, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)

    print "Performing grid search..."
    clf.fit(train[features], train["QuoteConversion_Flag"])

    # show scores and parameters for best run

    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    # create predictions on test data

    test_probs = clf.predict_proba(test[features])[:,1]

    sample = pd.read_csv(sample_sub_path)
    sample.QuoteConversion_Flag = test_probs
    sample.to_csv(test_prediction_file_string, index=False)

def train_one_model(train, test, features):
    # train a single model with one cross validation set 
    start = timer()
    # create instance of model 
    clf = xgb.XGBClassifier(n_estimators=1400,
                        max_depth=8,
                        learning_rate=0.025,
                        subsample           = 0.83, # 0.7
                        colsample_bytree    = 0.77, # 0.7
                        silent=True,
                        seed = 1400)
    # split the training data into training and validation sets 

    train_tr, train_val = train_test_split(train, test_size = 0.2)

    # cross validate model  
    start = time.clock()
    print "Cross validating model..."
    clf.fit(train_tr[features], train_tr["QuoteConversion_Flag"])
    print "Time to cross validate:", (time.clock() - start)/60., "minutes"

    # get validation score 
    val_predictions = clf.predict_proba(train_val[features])[:,1]
    val_correct = train_val.QuoteConversion_Flag.as_matrix()
    val_score = sk.metrics.roc_auc_score(val_correct, val_predictions)

    print "Cross val score: ", val_score

    if val_score > 0.966937: 
        print "New best validation AUC score! Old score: 0.966937" 
  
    # train model on all of the data before submitting test predictions
    print "Training model on all data..." 
    start = time.clock()
    clf.fit(train[features], train["QuoteConversion_Flag"]) 
    print "Time to train model:", (time.clock() - start)/60., "minutes"

    # put predictions on test data in output file
    test_predictions = clf.predict_proba(test[features])[:,1]
    sample = pd.read_csv(sample_sub_path)
    sample.QuoteConversion_Flag = test_predictions
    sample.to_csv(test_prediction_file_string, index=False)
    print "Wrote test predictions to file ", test_prediction_file_string
    print "Time to train model: ", timer() - start, " seconds"

def k_fold_cross_val_train_one_model(train, test, features, number_folds): 
    # one run that cross validates a model k-fold over training data 

    #############################################################################
    # Create model - this section is changed for each new model tried 
    #############################################################################

    clf = xgb.XGBClassifier(n_estimators=1300,
                        max_depth=8,
                        learning_rate=0.025,
                        subsample           = 0.83, # 0.7
                        colsample_bytree    = 0.77, # 0.7
                        silent=True,
                        seed = 1400)
    #############################################################################
    #############################################################################

    # set up cross validation split 
    results = []
    X, y = train[features], train["QuoteConversion_Flag"]
    cv = cross_validation.StratifiedKFold(y, n_folds=number_folds)

    # train model number_folds times on different splits of data and report results
    start = timer()
    for traincv, testcv in cv:
        predicted_probs = clf.fit(X.iloc[traincv], y.iloc[traincv]).predict_proba(X.iloc[testcv])[:,1]
        correct_target = y.iloc[testcv]
        AUC_score = sk.metrics.roc_auc_score(correct_target, predicted_probs)
        results.append(AUC_score)

    results = np.round_(results, decimals = 6)

    print "Cross validation AUC scores:"
    for _ in range(len(results)):
        print results[_]
    print "Average AUC:", np.round_(np.array(results).mean(),decimals = 6)
    print "AUC standard deviation:", np.round_(np.array(results).std(),decimals = 6)
    print "Time to perform cross validation:", (timer() - start)/60., "minutes" 

    # retrain model on all data and put predictions on test data in output file
    print "Training model on all available data..."
    test_predictions = clf.fit(train[features], train["QuoteConversion_Flag"]).predict_proba(test[features])[:,1]
    sample = pd.read_csv(sample_sub_path)
    sample.QuoteConversion_Flag = test_predictions
    sample.to_csv(test_prediction_file_string, index=False)
    print "Wrote test predictions to file ", test_prediction_file_string

def main(): 
    # main function that executes complete data cleaning and training pipeline
    print "Cleaning data..."
    train = clean_data(df)
    test = clean_data(dftest)

    print "Dropping useless columns..."
    train = drop_uniform_columns(train)
    test = drop_uniform_columns(test)

    print "Computing features..."
    train = add_new_features(train)
    test = add_new_features(test)

    features = list(train.columns[1:])  # first column is the conversion flag  

    #############################################################################
    # Training and validation 
    #############################################################################

    k_fold_cross_val_train_one_model(train, test, features, 5)

if __name__ == '__main__':
  main()