
# coding: utf-8

# In[ ]:


#This script takes in a binary-encoded matrix and outputs the PMI for each of the features 


# In[1]:


import pandas as pd
import numpy as np
import math
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import os
from pathlib import Path
#Progress bar
from tqdm import tqdm


# In[11]:


#def main():
# Load Data
DATA_DIR = "Data"
FEATURES_DIR = os.path.join(DATA_DIR, "retailFeatureSet-pmi.csv")
ENCODING_DIR = os.path.join(DATA_DIR, "binEncoding-PMI.csv")

fts = pd.read_csv(FEATURES_DIR)
binEnc = pd.read_csv(ENCODING_DIR)
binEnc=binEnc.drop('Unnamed: 0', axis = 1)

#Calculate PMI
posresults, negresults = CalcPMI(fts, binEnc)
#pmiposlist, pmineglist, pmidf = pmiForAllCal(fts, binEnc)

#Save PMI to a csv
#OUTPUT_DIR = os.path.join(DATA_DIR, "retailFeatureSet-10000.csv")
#pd.DataFrame.to_csv(fts, path_or_buf=OUTPUT_DIR)


# In[8]:


fts.head()


# In[10]:


def CalcPMI (fts, binEnc):
    LABEL = binEnc.columns[0]
    total = binEnc[LABEL].count()
    p_x = sum(binEnc[LABEL])/total
    p_x_0 = 1-p_x

    posresults = []
    negresults = []

    for ft in tqdm(fts['target_group']):
        if ft in binEnc.columns:
            p_y = sum(binEnc[ft])/total
            p_xy = sum(binEnc[ft][binEnc[LABEL]==1])/total
            if p_xy == 0:
                p_xy = 0.0001
            pmi = math.log(p_xy/(p_y*p_x),2)
            posresults.append([ft, pmi])
            
            p_y_0 = 1-p_y
            p_xy_0 = len(binEnc[(binEnc[LABEL]==0)&(binEnc[ft]==0)])/total
            if p_xy_0 == 0:
                p_xy_0 = 0.0001
            pmi = math.log(p_xy_0/(p_y_0*p_x_0),2)
            negresults.append([ft, pmi])
            

    posresults = pd.DataFrame(posresults)
    posresults.columns= ['target_group', 'pos_pmi']
    #fts = fts.set_index('target_group').join(posresults.set_index('target_group'))
    negresults = pd.DataFrame(negresults)
    negresults.columns= ['target_group', 'neg_pmi']
    #fts = fts.set_index('target_group').join(negresults.set_index('target_group'))
    return posresults, negresults
        
    


# In[14]:


posresults.sort_values('pos_pmi',ascending=0).head(15)


# In[15]:


negresults.sort_values('neg_pmi',ascending=0).head(15)


# In[60]:


def pmiIndivCal(df,x,gt, label_column='price_delta>.3'):
    px = sum(df[label_column]==gt)/len(df)
    py = sum(df[x]==1)/len(df)
    pxy = len(df[(df[label_column]==gt) & (df[x]==1)])/len(df)
    if pxy==0:#Log 0 cannot happen
        pmi = math.log((pxy+0.0001)/(px*py))
    else:
        pmi = math.log(pxy/(px*py))
    return pmi


# In[73]:


# Compute PMI for all terms and all possible labels
def pmiForAllCal(fts, df, label_column='price_delta>.3'):
    #Try calculate all the pmi for top k and store them into one pmidf dataframe
    pmilist = []
    pmiposlist = []
    pmineglist = []
    for word in tqdm(fts):
        #pmilist.append([word[0]]+[pmiCal(df,word[0])])
        pmiposlist.append([word[0]]+[pmiIndivCal(df,word[0],'1',label_column)])
        pmineglist.append([word[0]]+[pmiIndivCal(df,word[0],'0',label_column)])
    pmidf = pandas.DataFrame(pmilist)
    pmiposlist = pandas.DataFrame(pmiposlist)
    pmineglist = pandas.DataFrame(pmineglist)
    pmiposlist.columns = ['word','pmi']
    pmineglist.columns = ['word','pmi']
    #pmidf.columns = ['word','pmi']
    return pmiposlist, pmineglist, pmidf


# In[ ]:


pmiposlist, pmineglist, pmidf = pmiForAllCal(finaldf)

