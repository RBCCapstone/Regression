# Get features (stops words removed) by tokenizing corpus - no stemming in baseline
# Binary encoding
# Assign target group 
# Use mutual information to get final feature set
# baseline

import os
import re
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_selection import *
from tqdm import tqdm

# Testing Feature Selection
import nltk

## Download Resources
#nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

from nltk.stem import *

# download required resources
nltk.download("wordnet")

# we'll compare two stemmers and a lemmatizer
lrStem = LancasterStemmer()
sbStem = SnowballStemmer("english")
prStem = PorterStemmer()
wnLemm = WordNetLemmatizer()
def wnLemm_v(word):
    wnLemm = WordNetLemmatizer()
    word = wnLemm.lemmatize(word, 'v')
    return word


def assignStopWords(): 
    #Stop_words list Options
    #Variation 1: added stop words starting at 'one'
    stop_words = stopwords = [
        # dates/times
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "jan", "feb","mar", "apr", "jun", "jul", "aug", "oct", "nov", "dec", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning", "evening",
        # symbols that don't separate a sentence
        '$','“','”','’','—',
        # specific article terms that are useless
        "read", "share", "file", "'s","i", "photo", "percent","s", "t", "inc.", "corp", "group", "inc", "corp.", "source", "bloomberg", "cnbc","cnbcs", "cnn", "reuters","bbc", "published", "broadcast","york","msnbc",
        # other useless terms
        "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "co", "inc", "com", "theyve", "theyre", "theres", "heres", "didnt", "wouldn", "couldn", "didn","nbcuniversal","according", "just", "us", "ll", "times"#,
        # etc
        "from","the", "a", "with", "have", "has", "had", "having", "hello", "welcome", "yeah", "wasn", "today", "etc", "ext","definitely", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "while", "of", "said", "by", "for", "about", "into", "through", "during", "before", "after", "to", "from", "in", "out", "with", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "don", "now", "will"
        ]
    #from nltk.corpus import stopwords
    #stop_words = set(stopwords.words('english'))
    #print(stop_words)
    return stop_words


def corpus_count_words(df, stop_words, text_col = 'content', normalizer=None):
    tokenizer = RegexpTokenizer(r'\w+')
    word_counter = Counter()
    for row in df.itertuples(index=True, name='Pandas'):
            attribute = str((row, text_col))
            file_words = tokenizer.tokenize(attribute)
            #keep lowercased words that are not stop words as features
            file_wordsNS = [word.lower() for word in file_words if not word.lower() in stop_words]
            # remove words that are numbers
            file_wordsN = [word for word in file_wordsNS if not word.isnumeric()]
            #remove words with a word length less than 4 (i.e. 1-3)
            file_wordsF = [word for word in file_wordsN if not len(word)<4]
            
            #stem
            if normalizer:
                file_wordsF = [normalizer(word) for word in file_wordsF]
            
            word_counter.update(file_wordsF)
    return word_counter


#Binary encoding for features, also appends retail target group
def binary_encode_features(newsarticles, top_words, text_col = 'content', normalizer=None):
    tokenizer = RegexpTokenizer(r'\w+')
    df_rows = []
    for row in tqdm(newsarticles.itertuples(index=True, name='Pandas')):
            attribute = str((row, text_col))
            file_words = tokenizer.tokenize(attribute)
            if normalizer:
                file_words = [normalizer(word) for word in file_words]
            df_rows.append([1 if word.lower() in file_words else 0 for word in top_words])      
    X = pd.DataFrame(df_rows, columns = top_words)
    
    return X


def mutualInformation(B_Encoding, y, top_words): 
    #Estimate mutual information for a discrete target variable.
    #Mutual information (MI) [1] between two random variables is a non-negative value, which measures the dependency between the variables.
    #It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
    featureVals= mutual_info_classif(B_Encoding, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
    
    np.asarray(featureVals)

    Temp= pd.DataFrame(featureVals, columns = ['MI_Values'])
 
    Final = Temp.assign(target_group = top_words)
    
    Highest_Features = Final.nlargest(10000, 'MI_Values')
    
    return Highest_Features
#text_col = txtcol, norm = nrm, csv=True,

def selectFeatures(text_col = 'content', **kwargs):
    if ('articleDB' not in kwargs):
        df = importData()
    else:
        df = kwargs['articleDB']
        
    df.columns
    stop_words = assignStopWords()
    
    if ('norm' in kwargs):
        norm = kwargs['norm']
        normalizers = {'lrStem' : lrStem.stem,
                       'sbStem' : sbStem.stem,
                       'prStem' : prStem.stem,
                       'wnLemm' : wnLemm.lemmatize,
                       'wnLemm-v':wnLemm_v,
                       'baseline':None
                      }
        normalizer = normalizers[norm]
    
    #Select subset of orig data
    df1 = df[[text_col,'market_moving']]    
    news_cnt = corpus_count_words(df1, stop_words, text_col = text_col, normalizer = normalizer)
    
    print("starting Binary Encoding")
    num_features = 1000
    top_words = [word for (word, freq) in news_cnt.most_common(num_features)]
    B_Encoding = binary_encode_features(df1, top_words, text_col = text_col, normalizer = normalizer)
    y = df['market_moving']
    B_Encoding.assign(target_group=y)
      
    print("Finished Bin Encoding. Collecting Highest Features")
    Highest_Features = mutualInformation(B_Encoding, y, top_words)
    Highest_Features = pd.DataFrame(Highest_Features)
    
    # Save as csv file in DATACOLLECTION data folder (bc it's needed for encoding script)
    if ('csv' in kwargs) and (kwargs['csv']):
        
        # File path for this file
        file_name = text_col + '-' + norm + '-FeatureSet.csv'
        thispath = Path().absolute()
        OUTPUT_DIR = os.path.join(thispath, "Data", file_name)
        
        # if the following line throws an error, use the line after to save in same folder
        pd.DataFrame.to_csv(Highest_Features, path_or_buf=OUTPUT_DIR)
        #pd.DataFrame.to_csv(Highest_Features, path_or_buf=file_name)
    
    print(Highest_Features)
    return Highest_Features

def main(df):
    
    # options: 
    # norms = ['wnLemm-v', 'lrStem', 'sbStem', 'prStem', 'wnLemm']
    
    titleFts = selectFeatures(text_col = 'title', norm = 'wnLemm', articleDB = df, csv=True, )
    contentFts = selectFeatures(text_col = 'content', norm = 'wnLemm', articleDB = df, csv=True, )
    
    return titleFts, contentFts