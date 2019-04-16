
# coding: utf-8

# In[44]:


import nltk
from nltk.stem import *
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

import scipy.stats as stats
import sklearn
import random
import os
from pathlib import Path
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report 
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances, manhattan_distances, euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer

# we'll compare two stemmers and a lemmatizer
#lrStem = LancasterStemmer()
#sbStem = SnowballStemmer("english")
#wnLemm = WordNetLemmatizer()


# In[46]:


def getFeatures(ArticleDB):
    artText = ArticleDB["content"]
    countVect = CountVectorizer(binary=True)
    vector = countVect.fit(artText)
    features= countVect.vocabulary_
    fts = list(features.keys())
    
    return artText, fts


# In[48]:


#BinaryEncoding
def RecBinaryEncoding(fts, artText):
    print("bin Encoding")
    df_rows = []
    #tokenizer = RegexpTokenizer(r'\w+')

    for art in tqdm(artText):
        if type(art) == str: 
            body = art.lower()
            body = body.split() 
            wordsCounter = Counter(body)
            df_rows.append([1 if word in wordsCounter else 0 for word in fts])
        else:
            df_rows.append([0 for word in fts])
    X = pd.DataFrame(df_rows, columns = fts)

    return X


# In[49]:


#Term Freq. Encoding
def TfEncoding(fts, artText):
    print("tf Encoding")
    tf_rows = []
    
    for art in tqdm(artText):
        if type(art) == str:
            body = art.lower()
            body = body.split()
            wordsCounter = Counter(body)
            tf_rows.append([wordsCounter[word] if word in wordsCounter else 0 for word in fts])
        else:
            tf_rows.append([0 for word in fts])
    X = pd.DataFrame(tf_rows, columns = fts)
    
    return X


# In[61]:


#term Frequency - inverse document frequency encoding
def tfidfEncoding(fts, artText):
    print("tifidf Encoding")

    # Base calculations
    binX = RecBinaryEncoding(fts, artText)
    tfX = TfEncoding(fts, artText)
    
    # Calculate idf
    df_row = [binX[word].sum() for word in fts]
    idf = [1/(df+1) for df in df_row]
    #transpose list (not the cleverest method)
    idf_row = []
    idf_row.append(idf)
    idf_list = pd.DataFrame(idf_row, columns = fts)
    
    # Extract term frequencies
    tf = tfX.values
    # Set up loop to multiply each article (row) by the idf per term (col)
    tf_idf = []
    r, c = tf.shape
    for art in range(0,r):
        tf_idf.append(tf[art]*idf)
    tf_idf = pd.DataFrame(tf_idf, columns = fts)
    X = tf_idf
    
    return X


# In[51]:


def Cosinepairup(npV, rows, Y):
    for i in range(rows):
        #find most related articles indexed
        a = npV[i,:]
        index = np.argpartition(a, -4)[-4:]
        index2= index[np.argsort(a[index])]


        related = []
        #ensure that same article is not ranked as the most similar article
        for j in range(3,-1,-1):
            if i == index2[j]:
                pass #do not count the same article as most related
            elif len(related) == 3:
                pass
            else:
                related.append(str(index2[j]))

        Y.at[i, 'related_articles'] = ', '.join(related)

    return Y[['related_articles']]


# In[52]:


def recommender(ArticleDB):
    
    #Get Features
    artText, fts = getFeatures(ArticleDB)
    
    #Default encoding is tf-idf
    Encoded = tfidfEncoding(fts, artText)
    
    #Similarity matrix between each article
    Csim = cosine_similarity(Encoded)

    #convert to numpy
    npV = np.asarray(Csim)
    rows = np.size(npV,0)

    
    #match most related articles by article index
    finalMatches = Cosinepairup(npV, rows, Encoded)
    finalTable = ArticleDB.join(finalMatches, how='left')
    
    return finalTable

