
# coding: utf-8

# # Feature Encoding
# 
# The following script formats articles by features for linear regression (first attempt).  
# It takes in a list of features and a set of articles, converts to lowercase and creates an encoded matrix (dense)  
# While this isn't the cleverest method, it provides a usable input for setting up our initial linear regression code.
# 
# ### Limitations:
# * Proper Nouns should keep their capitals
# * Punctuation/Stemming etc not incorporated
# * Bi-grams not accommodated
# * Could be converted to sparse matrix
# * No log function incorporated at this point
#     
# 



#importing libraries
import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import os
from pathlib import Path


# Testing Word Normalization
import nltk

## Download Resources
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")


from nltk.stem import *

# download required resources
nltk.download("wordnet")

# we'll compare two stemmers and a lemmatizer
lrStem = LancasterStemmer()
sbStem = SnowballStemmer("english")
prStem = PorterStemmer()
wnLemm = WordNetLemmatizer()
def wnLemm_v(word):
    return WordNetLemmatizer.lemmatize(word, 'v')


def loadData(text_col, norm):
    DATA_DIR = "Data"
    feature_filename = text_col + '-' + norm + '-FeatureSet.csv'
    FEATURES_DIR = os.path.join(DATA_DIR, feature_filename)
    ARTICLES_DIR = os.path.join(DATA_DIR, "Labelled_Articles_.xlsx")
    
    fts = pd.read_csv(FEATURES_DIR)
    arts = pd.read_excel(ARTICLES_DIR)
    
    data = setData(fts, arts)
    
    return data
    
def setData(fts, arts, text_col):
    
    for col in fts.columns:
        if not (col.strip() == 'target_group'):
            fts = fts.drop([col], axis = 1)
    fts.columns = ['index']
    fts['index'] = list(map(lambda x: x.strip(), fts['index']))

    
    # Stripping out columns and only passing what we need
    artText = arts[text_col]
    data = {'fts':fts, 'artText': artText, 'article_id': arts['article_id'], 'market_moving':arts['market_moving']} #**
    return data



def binEncoding(data, normalizer=None):
    #print("Binary Encoding")
    fts = data['fts']
    artText = data['artText']
    df_rows = []
    tokenizer = RegexpTokenizer(r'\w+')

    for art in artText:
        if type(art) == str: 
            body = art.lower()
            #body = clean_file_text(body)
            art_words = tokenizer.tokenize(body)
            #insert word normalization
            if normalizer:
                art_words = [normalizer(word) for word in art_words]
            
            df_rows.append([1 if word in art_words else 0 for word in fts['index']])
        else:
            df_rows.append([0 for word in fts['index']])
    X = pd.DataFrame(df_rows, columns = fts['index'].values)
    return X



def tfEncoding(data, normalizer=None):
    print("tf Encoding")
    fts = data['fts']
    artText = data['artText']
    
    tf_rows = []
    for art in artText:
        if type(art) == str:
            body = art.lower()
            body = body.split()
            wordsCounter = Counter(body)
            tf_rows.append([wordsCounter[word] if word in wordsCounter else 0 for word in fts['index']])
        else:
            tf_rows.append([0 for word in fts['index']])
    X = pd.DataFrame(tf_rows, columns = fts['index'].values)  
    return X



def tfidfEncoding(data, normalizer=None):
    print("tifidf Encoding")
    fts = data['fts']

    # Base calculations
    binX = binEncoding(data)
    tfX = tfEncoding(data)
    
    # Calculate idf
    df_row = [binX[word].sum() for word in fts['index']]
    idf = [1/(df+1) for df in df_row]
    #transpose list (not the cleverest method)
    idf_row = []
    idf_row.append(idf)
    idf_list = pd.DataFrame(idf_row, columns = fts['index'])
    
    # Extract term frequencies
    tf = tfX.values
    # Set up loop to multiply each article (row) by the idf per term (col)
    tf_idf = []
    r, c = tf.shape
    for art in range(0,r):
        tf_idf.append(tf[art]*idf)
    tf_idf = pd.DataFrame(tf_idf, columns = fts['index'])
    X = tf_idf
    return X



def encoding(encodeType, text_col=None, norm=None, articles=None, features=None, **kwargs):
    # 0 for Binary Encoding
    # 1 for Term Frequency Encoding
    # 2 for TF-IDF Encoding
    # If you'd like to save as csv, use "csv = True"
    
    print('encoding ' + text_col + ' features')
    
    # Load up data
    if articles is None:
        data = loadData(text_col, norm)
    else:
        data = setData(features, articles, text_col)
    
    # Run corresponding encoding type and pass data
    options = {0 : binEncoding,
                1 : tfEncoding,
                2 : tfidfEncoding,}
    
    if norm:
        normalizers = {'lrStem' : lrStem.stem,
                       'sbStem' : sbStem.stem,
                       'prStem' : prStem.stem,
                       'wnLemm' : wnLemm.lemmatize,
                       'wnLemm-v':wnLemm_v,
                       'baseline':None
                      }
        normalizer = normalizers[norm]
    
    X = options[encodeType](data, normalizer)
    X = X.drop(['market_moving'], axis = 1) if 'market_moving' in list(X) else X
    
    # Append Y column and article ids
    Y = pd.DataFrame({'article_id':data['article_id'].values, 'market_moving':data['market_moving'].values})
    
    
    XY = Y.join(X)
    
    # Save as csv file 
    if ('csv' in kwargs) and (kwargs['csv']):
        
        # File path for this file
        file_name = text_col +'-'+ norm +'-'+  options[encodeType].__name__ + '.csv'
        thispath = Path().absolute()

        # if the following line throws an error, use the line after to save in same folder
        OUTPUT_DIR = os.path.join(thispath, "Data", file_name)
        pd.DataFrame.to_csv(XY, path_or_buf=OUTPUT_DIR)
        #pd.DataFrame.to_csv(XY, path_or_buf=file_name)
    
    # Return Panda DataFrame
    return XY
    


def main(titleFts, contentFts, articleDB): # Stuff to do when run from the command line    
    # norm options: ['lrStem', 'sbStem', 'prStem', 'wnLemm', 'wnLemm-v']
    
    title_enc = encoding(0, text_col = 'title', norm = 'wnLemm', features = titleFts, articles = articleDB, csv = True)
    content_enc = encoding(0, text_col = 'content', norm = 'wnLemm', features = contentFts, articles = articleDB, csv = True)
    
    return title_enc, content_enc
