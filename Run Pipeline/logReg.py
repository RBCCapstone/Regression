import pandas as pd
import numpy as np
import scipy.stats as stats
import sklearn
import random
import os
from pathlib import Path
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report 
import pickle


def LoadData(filename):
    DATA_DIR = "Data"
    ENCODING_DIR = os.path.join(DATA_DIR, filename)
    data = pd.read_csv(ENCODING_DIR)
    return data

def runLogReg(titleMx, contentMx, articleDB):
    #X = LoadData(filename) # This would be named to whatever today's binEncoding file is called
    body_fts = 350
    title_fts = 30 # number of features to use for titles
    title_w = 5      # weight of features to use for titles
    
    X = contentMx.drop(list(contentMx)[body_fts-1:], axis=1)
    #artID = X['article_id']
    artID = articleDB['url']
    try:
        X = X.drop(columns=['article_id'])
    except:
        pass
    
    # Combine title and body feature sets as desired

    # Load the data, keep desired number of title features, multiply them by desired weight, then join to data
    data_title = titleMx
    #data_title = data_title.drop(columns= ['Unnamed: 0'])        
    data_title = data_title.drop(list(data_title)[title_fts:], axis=1)
    data_title = data_title*title_w

    X = X.join(data_title, rsuffix = '-title')

    #print(X.head(20))
    classifier = pickle.load(open("ourClassifier.p", "rb"))
    
    y_predict = classifier.predict(X)
    # get log scores for train and test set
    y_proba = classifier.predict_proba(X)    
    
    #tie the scores and predictions to specific articles
    scores = pd.DataFrame(data=y_proba)
    scores['article_id'] = artID.values
    scores['prediction'] = y_predict
    
    #rename Columns
    #print(scores.columns)
    scores.columns = ['nonRel', 'Rel', 'url', 'prediction']
    
    #rank articles from most to least relevant
    ScoreRanked = scores.sort_values(['Rel'], ascending=[0])

    #import cleaned articles
    #DATA_DIR = "Data"
    #ENCODING_DIR = os.path.join(DATA_DIR, 'cleanedArticles.xlsx')
    Articles = articleDB
    
    
    #Do a sql style join with the results and actual article data on 'article_id'   
    Combined = ScoreRanked.merge(Articles, on='url', how='left')
    #print(Combined[['title', 'description']].head())
    #Save as csv, tells you encoding type
    #thispath = Path().absolute()
    #OUTPUT_DIR = os.path.join(thispath, "Data", "results_"+filename)
    #pd.DataFrame.to_csv(Combined, path_or_buf=OUTPUT_DIR)
    
    #Save as excel file (better because weird characters encoded correctly)
    #OUTPUT_DIR = os.path.join(DATA_DIR, "results_encoding.xlsx")
    #writer = pd.ExcelWriter(OUTPUT_DIR)
    #Combined.to_excel(writer,'Sheet1')
    #writer.save()
    return Combined

