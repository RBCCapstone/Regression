#Script to extract important topics from content
# based on code written by: vipul-sharma20
# modifications made by: jadekhiev

# imports
import os
import sys
from pathlib import Path

# imports required utility functions
import string
from collections import Counter

# Data packages
import math
import pandas as pd
import numpy as np

#Operation
import operator

#Natural Language Processing Packages
import re
import spacy
# python -m spacy download en
try:
    nlp = spacy.load('en') #spacy PoS tagger
except:
    import en_core_web_sm
    nlp = en_core_web_sm.load()

#Progress bar
from tqdm import tqdm

# Utility functions for context extraction
def tagWords(article):
    # spacy context extraction
    # this is our spacy tagger 
    taggedArticle = nlp(article)
    taggedTerm = []
    stopwords = [
        # dates/times
          "january", "february", "march", "april", "may", "june", "july", "august", "september", "october"
        , "november", "december", "jan", "feb","mar", "apr", "jun", "jul", "aug", "oct", "nov", "dec"
        , "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning","evening"
        ,"today","pm","am","daily" 
        # specific article terms that are useless
        , "read", "file", "'s","'t", "photo", "inc", "corp", "group", "inc", "corp", "source"
        , "bloomberg", "cnbc","cnbcs", "cnn", "reuters","bbc", "published", "broadcast","msnbc","ap"
        , "said","nbcuniversal","newsletterupgrade","nbc", "news",'url',"cbc"
        # other useless terms
        , "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself"
        , "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its"
        , "itself", "they", "them", "their", "theirs","themselves", "what", "which", "who", "whom"
        , "this", "that", "these", "those", "theyve", "theyre", "theres", "heres", "didnt", "wouldn"
        , "couldn", "didn","are","is", "was","will", "have", "be", "such","did","put"
        , "mr", "mr.", "ms", "ms.","mrs", "mrs."
    ]
    for token in taggedArticle:
        if token.text.lower() not in stopwords:
            if len(token.text)>2:
                taggedTerm.append((token.text,token.pos_,token.dep_))
            else: # collect numbers and symbols (percents, dollar signs, etc.)
                if token.text.isdigit() or token.text in ('%'): taggedTerm.append((token.text,token.pos_,token.dep_))
            
    return taggedTerm

def countWords(wordList):
    return dict(Counter(wordList))

def getContextTags(content):
    taggedTerm = tagWords(content)
    normalized = True
    while normalized:
        normalized = False
        for i in range(0, len(taggedTerm) - 1):
            token_1 = taggedTerm[i]
            if i+1 >= len(taggedTerm) - 1:
                break
            token_2 = taggedTerm[i+1]

            # chunk nouns
            if token_1[1] in ('NOUN','PROPN') and token_1[2]=='compound' and token_2[1]!='PUNCT':
                newTerm = taggedTerm[i][0]+" "+taggedTerm[i+1][0]
                pos = taggedTerm[i+1][1]
                dep = taggedTerm[i+1][2]
                taggedTerm.insert(i+2, (newTerm, pos, dep))
                taggedTerm.pop(i) # remove word 1
                taggedTerm.pop(i) # remove word 2
                normalized = True

            # chunk nouns with their adjectives
            elif token_1[1]=='ADJ' and token_2[1] in ('NOUN','PROPN'):
                newTerm = taggedTerm[i][0]+" "+taggedTerm[i+1][0]
                pos = taggedTerm[i+1][1]
                dep = taggedTerm[i+1][2]
                taggedTerm.insert(i+2, (newTerm, pos, dep))
                taggedTerm.pop(i) # remove word 1
                taggedTerm.pop(i) # remove word 2
                normalized = True

            # capture nouns that are composed of verb + noun (e.g. share price)
            elif token_1[1]=='VERB' and token_1[2] in ('ccomp') and token_2[1]=='NOUN':
                newTerm = taggedTerm[i][0]+" "+taggedTerm[i+1][0]
                pos = taggedTerm[i+1][1]
                dep = taggedTerm[i+1][2]
                taggedTerm.insert(i+2, (newTerm, pos, dep))
                taggedTerm.pop(i) # remove word 1
                taggedTerm.pop(i) # remove word 2
                normalized = True        

            # chunk hyphenated words
            elif token_1[2] in ('compound','npadvmod','amod','advmod','nmod','intj') and token_2[0]=='-':
                newTerm = taggedTerm[i][0]+taggedTerm[i+1][0]+taggedTerm[i+2][0]
                pos = 'ADJ'
                dep = 'amod'
                taggedTerm.insert(i+3, (newTerm, pos, dep))
                taggedTerm.pop(i) # remove word 1
                taggedTerm.pop(i) # remove word 2
                taggedTerm.pop(i) # remove word 3
                normalized = True

            # chunk numeric terms like money and percents
            elif token_1[1] in ('NUM','SYM','NVAL') and token_1[2] in ('nmod','nummod','quantmod','compound'):
                if token_1[1] in ('NVAL') and token_2[1] == 'NOUN' and token_2[2] != 'pobj':
                    break
                elif token_1[1] in ('SYM'):
                    newTerm = taggedTerm[i][0]+taggedTerm[i+1][0]
                    pos = 'NVAL' # number val
                    dep = taggedTerm[i+1][2]
                    taggedTerm.insert(i+2, (newTerm, pos, dep))
                    taggedTerm.pop(i) # remove word 1
                    taggedTerm.pop(i) # remove word 2
                    normalized = True
                else:
                    newTerm = taggedTerm[i][0]+" "+taggedTerm[i+1][0]                  
                    pos = 'NVAL' # number val
                    dep = taggedTerm[i+1][2]
                    taggedTerm.insert(i+2, (newTerm, pos, dep))
                    taggedTerm.pop(i) # remove word 1
                    taggedTerm.pop(i) # remove word 2
                    normalized = True

    highlight_text = []
    noun_phrases = []
    for token in taggedTerm:
        term = token[0]
        pos = token[1]
        dep = token[2]
        if pos in ('NOUN', 'PROPN') and dep not in ('npadvmod','amod','advmod'):
            if not(pos == 'NOUN' and len(term.split())<2):
                highlight_text.append(term)
                noun_phrases.append(term)
        elif pos in ('NVAL'): # highlight number values
            highlight_text.append(term)
    
    return highlight_text, noun_phrases

# extract all unigrams based on all words pulled from context extraction
def unigramBreakdown(fullContext):
    stopwords = [
    # dates/times
      "january", "february", "march", "april", "may", "june", "july", "august", "september", "october"
    , "november", "december", "jan", "feb","mar", "apr", "jun", "jul", "aug", "oct", "nov", "dec"
    , "jan.", "feb.","mar.", "apr.", "jun.", "jul.", "aug.", "oct.", "nov.", "dec."
    , "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning","evening"
    , "today","pm","am","daily","day", "year"
    # specific article terms that are useless
    , "read", "file", "n't","'s","'t", "photo", "inc", "corp", "group", "inc", "corp", "source"
    , "bloomberg", "cnbc","cnbcs", "cnn", "reuters","bbc", "published", "broadcast","msnbc","ap"
    , "said","nbcuniversal","newsletterupgrade","nbc", "news",'url', "more information","cbc"
    , 'business insider', 'new york times', "wall street journal","washington post"
    # other useless terms
    , "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself"
    , "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its"
    , "itself", "they", "them", "their", "theirs","themselves", "what", "which", "who", "whom"
    , "this", "that", "these", "those", "theyve", "theyre", "theres", "heres", "didnt", "wouldn"
    , "couldn", "didn","are","is", "was","will", "have", "be", "were"
    , "company", "people", "president", "others", "times", "percent","number", "companies", "business"
    , "world", "state", "order","talk",'team', 'brands', 'program'
    , 'family', 'everyone', 'per', 'house', 'case', 'someone', 'something', 'anyone',"person"
    , "co.", "co", "inc.", "inc", ".com", "com", "report", "things", "thing", "job", "member", "members"
    , "staying", "possibility","part", "none","showing", "one"
    , "us", "u.s", "u.s.", "united states", "america", "americans", "united states of america", "usa", "states"
    ]
    
    # separates each word for each article => list of list
    articleUnigrams = []
    for term in fullContext:
        articleUnigrams.extend(term.split())
    
    # remove stop words and punctuation
    translator = str.maketrans('', '', string.punctuation)
    unigrams = [term.lower().translate(translator) for term in articleUnigrams if term.lower() not in stopwords and len(term)>2]
    # count frequency of terms
    # unigrams = countWords(unigrams)  
    return unigrams

# extracts unigrams AND bigrams pulled by context extraction
def bigramBreakdown(fullContext):
    stopwords = [
    # dates/times
      "january", "february", "march", "april", "may", "june", "july", "august", "september", "october"
    , "november", "december", "jan", "feb","mar", "apr", "jun", "jul", "aug", "oct", "nov", "dec"
    , "jan.", "feb.","mar.", "apr.", "jun.", "jul.", "aug.", "oct.", "nov.", "dec."
    , "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning","evening"
    , "today","pm","am","daily","day", "year"
    # specific article terms that are useless
    , "read", "file", "'s","'t", "photo", "inc", "corp", "group", "inc", "corp", "source"
    , "bloomberg", "cnbc","cnbcs", "cnn", "reuters","bbc", "published", "broadcast","msnbc","ap"
    , "said","nbcuniversal","newsletterupgrade","nbc", "news",'url', "more information","cbc"
    , 'business insider', 'new york times', "wall street journal","washington post"
    # other useless terms
    , "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself"
    , "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its"
    , "itself", "they", "them", "their", "theirs","themselves", "what", "which", "who", "whom"
    , "this", "that", "these", "those", "theyve", "theyre", "theres", "heres", "didnt", "wouldn"
    , "couldn", "didn","are","is", "was","will", "have", "be", "were"
    , "company", "people", "president", "others", "times", "percent","number", "companies", "business"
    , "world", "state", "order","talk",'team', 'brands', 'program'
    , 'family', 'everyone', 'per', 'house', 'case', 'someone', 'something', 'anyone',"person"
    , "co.", "co", "inc.", "inc", ".com", "com", "report", "things", "thing", "job", "member", "members"
    , "staying", "possibility","part", "none","showing", "one"
    , "us", "u.s.", "united states", "america", "united states of america", "usa", "states"
    ]
    bigrams = []
    # remove punctuation and translate all terms into lowercse
    translator = str.maketrans('', '', string.punctuation)
    #bigrams.extend([term.lower().translate(translator) for term in fullContext if len(term.split()) < 3 and term.lower not in stopwords])
    bigrams.extend([term.lower() for term in fullContext if len(term.split()) < 3 and term.lower() not in stopwords])
    
    return bigrams

# did this because I couldn't good way to write the switcher to switch to a non-function
def ngramBreakdown(keyterms):
    stopwords = [
    # dates/times
      "january", "february", "march", "april", "may", "june", "july", "august", "september", "october"
    , "november", "december", "jan", "feb","mar", "apr", "jun", "jul", "aug", "oct", "nov", "dec"
    , "jan.", "feb.","mar.", "apr.", "jun.", "jul.", "aug.", "oct.", "nov.", "dec."
    , "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning","evening"
    , "today","pm","am","daily","day", "year"
    # specific article terms that are useless
    , "read", "file", "'s","'t", "photo", "inc", "corp", "group", "inc", "corp", "source"
    , "bloomberg", "cnbc","cnbcs", "cnn", "reuters","bbc", "published", "broadcast","msnbc","ap"
    , "said","nbcuniversal","newsletterupgrade","nbc", "news",'url', "more information","cbc"
    , 'business insider', 'new york times', "wall street journal","washington post"
    # other useless terms
    , "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself"
    , "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its"
    , "itself", "they", "them", "their", "theirs","themselves", "what", "which", "who", "whom"
    , "this", "that", "these", "those", "theyve", "theyre", "theres", "heres", "didnt", "wouldn"
    , "couldn", "didn","are","is", "was","will", "have", "be", "were"
    , "company", "people", "president", "others", "times", "percent","number", "companies", "business"
    , "world", "state", "order","talk",'team', 'brands', 'program'
    , 'family', 'everyone', 'per', 'house', 'case', 'someone', 'something', 'anyone',"person"
    , "co.", "co", "inc.", "inc", ".com", "com", "report", "things", "thing", "job", "member", "members"
    , "staying", "possibility","part", "none","showing", "one"
    , "us", "u.s.", "united states", "america", "united states of america", "usa", "states"
    ]
    ngrams = []
    # remove punctuation and translate all terms into lowercse
    # translator = str.maketrans('', '', string.punctuation)
    #bigrams.extend([term.lower().translate(translator) for term in fullContext if len(term.split()) < 3 and term.lower not in stopwords])
    ngrams.extend([term.lower() for term in keyterms if term.lower() not in stopwords])
    
    return ngrams

# PMI For Tag Ranking
# return binary representation of article in terms of all keyphrases pulled
def dfTransform(df, term_column):
    # df is the article df ;
    keyterms = []
    for article in df[term_column].values:
        keyterms.extend([word.lstrip() for word in (article.split(','))])
    keyterms = set(keyterms) # deduplicate terms by casting as set
    
    # for each article and each keyword: give 1 if keyword in article and 0 if not
    encodedArticle = []
    for i in tqdm(df.index):
        articleTerms = ([word.lstrip() for word in (df[term_column].iloc[i].split(','))])
        encodedArticle.append([1 if word in articleTerms else 0 for word in keyterms])
    
    # set up dataframe
    binEncDf = pd.DataFrame(encodedArticle)
    # use keywords as columns
    binEncDf.columns = keyterms
    # keep article_id and prediction from original table
    df = df.rename(columns={'prediction': 'mkt_moving'}) # changed it from prediction because that was also a keyterm
    # join prediction with encoding
    binEncDf = df[['mkt_moving']].join(binEncDf)
    
    return binEncDf

# Simple example of getting pairwise mutual information of a term
def pmiCal(df, x, label_column='mkt_moving'):
    pmilist=[]
    for i in [0,1]:
        for j in [0,1]:
            px = sum(df[label_column]==i)/len(df)
            py = sum(df[x]==j)/len(df)
            pxy = len(df[(df[label_column]==i) & (df[x]==j)])/len(df)
            if pxy==0:#Log 0 cannot happen
                pmi = math.log((pxy+0.0001)/(px*py+0.0001))
            else:
                pmi = math.log(pxy/(px*py+0.0001))
            pmilist.append([i]+[j]+[px]+[py]+[pxy]+[pmi])
    pmiDf = pd.DataFrame(pmilist)
    pmiDf.columns = ['x','y','px','py','pxy','pmi']
    
    return pmiDf

def pmiIndivCal(df,x,gt, label_column='mkt_moving'):
    px = sum(df[label_column]==gt)/len(df)
    py = sum(df[x]==1)/len(df)
    pxy = len(df[(df[label_column]==gt) & (df[x]==1)])/len(df)
    if pxy==0:#Log 0 cannot happen
        pmi = math.log((pxy+0.0001)/(px*py+0.0001))
    else:
        pmi = math.log(pxy/(px*py))
    
    return pmi

# calculate all the pmi for all tags across all articles and store top 5 tags for each article in df
def pmiForAllCal(artDf, binaryEncDf, term_column, label_column='mkt_moving'): 
    
    for i in tqdm(artDf.index): # for all articles
        terms = set(([word.lstrip() for word in (artDf[term_column].iloc[i].split(','))]))
        pmiList = []

        for word in terms:
            pmiList.append([word]+[pmiIndivCal(binaryEncDf,word,1,label_column)])
        
        pmiList = pd.DataFrame(pmiList)
        pmiList.columns = ['word','pmi']
        artDf.at[i,'tags_top_5'] = (',').join(word for word in pmiList.sort_values(by='pmi', ascending=False).head(5)['word'])    
    return artDf

# Functions to run extraction and rank tags

# Tag ranking using PMI
def calculatePMI(artDf, termType):
    # use PMI to calculate top 10 terms that should be displayed for each article    
    # get binary encoding of articles represented as uni- and bigrams
    binaryEncDf = dfTransform(artDf, termType)
    articleDf_ranked = pmiForAllCal(artDf, binaryEncDf, termType)
    
    return articleDf_ranked, binaryEncDf

# find most popular keyterms mentioned in news
def frequencyCounter(binEncDf):
    binEncDf = binEncDf.drop(['mkt_moving'], axis=1)
    # sum each column of binary encoded articles
    # output should be a dataframe with: word | # of articles mentioning word
    freqDf = binEncDf.sum(axis=0, skipna=True).sort_values(ascending=False).to_frame().reset_index()
    freqDf.columns = ['word','freq_articles']
    
    return freqDf

# Retrieve context
def retrieveContext(articleDB, termType='ngrams'):
    # import classified articles
    articleDf = articleDB
    
    breakdown = {
        'ngrams': ngramBreakdown, # store n-grams pulled from context extraction
        'bigrams': bigramBreakdown, # store bigrams and unigrams captured by context extraction
        'unigrams': unigramBreakdown # store unigrams captured by separating all terms pulled by context extraction
        }
    
    for i in articleDf.index:
        # get context for articles
        fullContext, keyTerms = getContextTags(articleDf['contentWithStops'].iloc[i])
        articleDf.at[i, 'context'] = ', '.join(fullContext) # highlight these terms within article 
        articleDf.at[i, 'tags'] = ', '.join(breakdown[termType](keyTerms)) # use these as tags as they are limited to noun/noun phrases
    
    # returns article Df with new column for top tags
    articleDf, binaryEncDf = calculatePMI(articleDf, 'tags')
    
    # returns most popular terms mentioned across all articles
    trendingTermsDf = frequencyCounter(binaryEncDf)

    return articleDf, trendingTermsDf