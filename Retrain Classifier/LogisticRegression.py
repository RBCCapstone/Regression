# This script runs the logistic regression and tunes parameters to output a new classifier (pickle)


# Importing modules
get_ipython().run_line_magic('matplotlib', 'inline')
#rom __future__ import print_function
import pandas as pd
import numpy as np
import scipy.stats as stats
import sklearn
sklearn.warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import random
import os
from pathlib import Path
from sklearn.linear_model import *
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report 
from sklearn.model_selection import GridSearchCV
import pickle



def LoadSplitData(split, body_file,  body_fts, title_file=None, title_fts=None, title_w=None):
    
    if isinstance(body_file, str):
        data_xy = LoadData(body_file)
    else:
        data_xy = body_file

    y = data_xy['market_moving']
    data_xy = data_xy.drop(columns= ['market_moving'])
    
    # separates out X & Y columns, but keeps article ID with X for split
    # num_features = 300
    data_xy = data_xy.drop(list(data_xy)[body_fts+1:], axis=1)
    
    # Combine title and body feature sets as desired
    if isinstance(title_file, str):
    # Load the data, keep desired number of title features, multiply them by desired weight, then join to data
        if isinstance(title_file, str):
            data_title = LoadData(title_file)
        else:
            data_title = title_file
            
        data_title = data_title.drop(columns= ['market_moving', 'article_id'])
        data_title = data_title.drop(list(data_title)[title_fts:], axis=1)
        data_title = data_title*title_w
        
        data_xy = data_xy.join(data_title, rsuffix = '-title')
    
    
    #Get dummy test/train set 
    DummyX_train, DummyX_test, Dummyy_train, Dummyy_test = train_test_split(data_xy, y, test_size=split, random_state=42)
    print(data_xy.shape)
    return  data_xy

def LoadData(filename):
    DATA_DIR = "Data"
    ENCODING_DIR = os.path.join(DATA_DIR, filename)
    data = pd.read_csv(ENCODING_DIR)
    data = data.drop(columns= ['Unnamed: 0'])
    return data



def all_metrics(y, y_hat):
    scores={}
    # takes in the actual score (y) and the prediction (y_hat)
    scores['average_precision_score'] = average_precision_score(y, y_hat)
    scores['accuracy_score']= accuracy_score(y, y_hat)
    scores['precision_score']= precision_score(y, y_hat)
    scores['recall_score'] = recall_score(y, y_hat)
    scores['f1_score'] =  f1_score(y, y_hat)
    scores['confusion_matrix'] = confusion_matrix(y, y_hat)
    scores['classification_report'] = classification_report(y, y_hat)
    return scores

def SingleTest(X_train, y_train, X_test, y_test, penaltyval, Cval):
    #Use Logistic Regression - Testing with dummy-y-variable
    
    #extract Article IDs
    trainID = X_train['article_id']
    testID = X_test['article_id']
    X_train = X_train.drop(columns=['article_id'])
    X_test = X_test.drop(columns=['article_id'])
    
    #define classifier
    ##penaltyval = 'l2'
    logReg = LogisticRegression(penalty=penaltyval, dual=False, tol=0.0001, C=Cval, fit_intercept=True, random_state=0, solver='liblinear')

    #Correction?? Build the classifier
    clfSingleTest = logReg.fit(X_train, y_train)
    # Save the classifier
    pickle.dump(clfSingleTest, open("ourClassifier.p", "wb"))

    # predict on train and test set
    y_train_predict = clfSingleTest.predict(X_train)
    y_test_predict = clfSingleTest.predict(X_test)
    
    # get log scores for train and test set
    y_train_log_scores = clfSingleTest.predict_log_proba(X_train)
    y_test_log_scores = clfSingleTest.predict_log_proba(X_test)
    
    
    
    #tie the scores and predictions to specific articles
    train_scores = pd.DataFrame(data=y_train_log_scores)
    train_scores['article_id'] = trainID.values
    train_scores['prediction'] = y_train_predict
    test_scores = pd.DataFrame(data=y_test_log_scores)
    test_scores['article_id'] = testID.values
    test_scores['prediction'] = y_test_predict

    ## Calculate Binary metrics
    columns = ['Precision','Recall', 'F1', 'Avg Precision', 'Accuracy']
    df = pd.DataFrame(index=['Train','Test'], columns=columns)
    
    TrainPrecision = precision_score(y_train, y_train_predict)
    TestPrecision = precision_score(y_test, y_test_predict)
    
    TrainRecall = accuracy_score(y_train, y_train_predict)
    TestRecall = accuracy_score(y_test, y_test_predict)
    
    Trainf1 = f1_score(y_train, y_train_predict, average='binary')
    Testf1 = f1_score(y_test, y_test_predict, average='binary')
    
    ## Calculate all metrics
    all_train_scores = all_metrics(y_train, y_train_predict)
    all_test_scores = all_metrics(y_test, y_test_predict)
    
    
    #Not to be confused with the ranking metric, mAP (mean average precision), this is simply the average of the P and R curve
    TrainAvgP = average_precision_score(y_train, y_train_predict)
    TestAvgP = average_precision_score(y_test, y_test_predict)
    
    TrainAccuracy = accuracy_score(y_train, y_train_predict)
    TestAccuracy = accuracy_score(y_test, y_test_predict)
    
    df.loc['Train'] = pd.Series({'Precision': TrainPrecision, 'Recall': TrainRecall, 'F1': Trainf1, 'Avg Precision': TrainAvgP, 'Accuracy': TrainAccuracy})
    df.loc['Test'] = pd.Series({'Precision': TestPrecision, 'Recall': TestRecall, 'F1': Testf1, 'Avg Precision': TestAvgP, 'Accuracy': TestAccuracy})
    return df, train_scores, test_scores, all_train_scores, all_test_scores



def SequentialSetRun(X, y, testsize):
   
    #Predicting on Real DataSet - Only 1 run
    num_articles = len(X) -1 #Subtract header row
    #testsize = 0.30
    trainsize = 1-testsize


    #Select first 70% as train
    X_train = X.iloc[:round(num_articles*trainsize)]
    y_train = y.iloc[:round(num_articles*trainsize)]

    #Following testsize (30% default) is test
    X_test = X.iloc[(round(num_articles*trainsize)):]
    y_test = y.iloc[(round(num_articles*trainsize)):]
   
    #Run SingleTest
    TestResults, train_scores, test_scores = SingleTest(X_train, y_train, X_test, y_test)

    return TestResults, train_scores, test_scores



def runLogReg(filename):
    X = LoadData(filename) # This would be named to whatever today's binEncoding file is called
    artID = X['article_id']
    X = X.drop(columns=['article_id'])
    #todo:extra cols
    xcols = [0,1,2,3,4,5,6]
    X = X.drop(X.columns[xcols], axis=1)
    print(X.head())
    classifier = pickle.load(open("ourClassifier.p", "rb"))
    
    y_predict = classifier.predict(X)
    # get log scores for train and test set
    y_log_proba = classifier.predict_log_proba(X)    
    
    #tie the scores and predictions to specific articles
    scores = pd.DataFrame(data=y_log_proba)
    scores['article_id'] = artID.values
    scores['prediction'] = y_predict
    
    thispath = Path().absolute()
    OUTPUT_DIR = os.path.join(thispath, "Data", "results_"+filename)
    pd.DataFrame.to_csv(scores, path_or_buf=OUTPUT_DIR)

#------------------------------------
# #  kfold CV with sequential splits
#------------------------------------


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def getHyperparameters(X,y):
    #create regularistcation penalty space
    penalty = ['l1','l2'] #only l2 for now
    
    #Create regularization hyperparameter space
    C = [ 0.01, 0.1, 1, 10]
    #C = np.logspace(0,4,10)

    
    #create hyperparemeter options
    parameters = dict(C=C, penalty=penalty)

    logistic = LogisticRegression()
    clf = GridSearchCV(logistic, parameters)


    best_model = clf.fit(X,y) #default 3 cross validation default
    
    # View best hyperparameters
    BestPenalty = best_model.best_estimator_.get_params()['penalty']
    BestC = best_model.best_estimator_.get_params()['C']
    
    print('Best Penalty:', BestPenalty, 'Best C:', BestC)
    
    return BestPenalty, BestC



def run_from_start(body_file,  body_fts, title_file=None, title_fts=None, title_w=None):
    # Setting Up Experiments
    data_xy = LoadSplitData(0.3, body_file, body_fts, title_file, title_fts, title_w)

    #kFold Cross Validation using Day Forward-Chaining
    #We want to split the data into sequential folds
    from sklearn.model_selection import GridSearchCV

    #let k = # of folds to test on
    k = 5

    #y column index (set where the y value is located)
    yindex = 1

    #Checking indexing
    newdata = data_xy
    print(newdata.index)
    train_index = int(len(newdata) / k)


    #Nested KFold Cross Validation - Prints Precision value and informs of each hyperparam used for each outerfold
    testPrec = []
    testAcc = []
    testRecall=[]
    testScores=[]
    trainScores=[]
    for i in range(k): 
        #Get indexes for test and train data for split i in k
        incrementrows = int(len(newdata) / (k+1))

        train_index_start = 0
        train_index_end = train_index_start + (incrementrows * (i+1))
        test_index_start = train_index_end + 1

        #if it's the last iteration, add leftover articles to test set - (due to rounding)
        if i == (k-1):
            test_index_end = int(len(newdata))
        else:
            test_index_end = test_index_start + incrementrows

        #print(i, train_index_start, train_index_end, test_index_start, test_index_end)

        #Extract the train/validation split
        trainsplitsubset = data_xy.iloc[train_index_start:train_index_end]

        #dropping first column because trainsplitsubset has the y value
        Xtrain = trainsplitsubset.drop(trainsplitsubset.columns[yindex],axis=1)
        ytrain = trainsplitsubset.iloc[:,yindex] 

        #train/validate with GridSearchCV to get Hyperparameters first
        Penalty, C = getHyperparameters(Xtrain,ytrain)
        #print(C)

        #Extract the test set
        testsplitsubset = data_xy.iloc[test_index_start:test_index_end]
        Xtest = testsplitsubset.drop(testsplitsubset.columns[yindex], axis=1)
        ytest = testsplitsubset.iloc[:,yindex]

        #print(Xtest.head())
        #print(ytest.head())

        #Use these hyperparamers on outerfold
        df, train_scores, test_scores, all_train_scores, all_test_scores = SingleTest(Xtrain, ytrain, Xtest, ytest, Penalty, C)

        #focusing on precision (can access TestResults1 at different indices to evaluate more metrics)
        testPrec.append(df.iloc[1,0])
        testAcc.append(df.iloc[1,4])
        #print(testAcc)

        testScores.append(all_test_scores)
        trainScores.append(all_train_scores)

        #Print mean Precision score (average binary precision over k outer folds)
        meanPrecision = sum(testPrec) / float(len(testPrec))

        #Print Accuracy
        meanAccuracy = sum(testAcc) / float(len(testAcc))

        if i == k-1:
            print(meanPrecision)
            print(meanAccuracy)
    return testScores, trainScores


def main(body_file=None, title_file=None ):
    # norm options = ['lrStem', 'sbStem', 'prStem', 'wnLemm', 'wnLemm-v']
    results_df = pd.DataFrame()
    #for nrm in nrms:
    nrm = 'wnLemm'
    content_ft = 350
    title_ft = 30
    title_weight = 5

    if body_file is None:
        body_file = 'content-'+nrm+'-binencoding.csv'
    if title_file is None:
        title_file = 'title-'+nrm+'-binencoding.csv'

    #run the thing and output testScores
    testScores, trainScores = run_from_start(body_file, content_ft, title_file, title_ft, title_weight)

    #    x, y, xy_data = LoadSplitData(0.3, body_file, content_ft, title_file, title_ft, title_weight)

    df = pd.DataFrame(testScores)
    df['normalizer'] = nrm
    df['content_features'] = content_ft
    df['title_features'] = title_ft
    df['title_weight'] = title_weight

    results_df = results_df.append(df)
    
    return results_df