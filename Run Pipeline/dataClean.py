import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pandas as pd
import re, string

def DataClean(articleDf):
    #remove blanks (NaN)
    articleDf = articleDf.dropna(subset = ['content', 'title']) 

    #remove blocked articles without content
    articleDf = articleDf[articleDf.content.str.contains('Your usage has been flagged', case=False) == False]
    articleDf = articleDf[articleDf.content.str.contains('To continue, please click the box', case=False) == False]

    # vidoes/ads/commentary
    articleDf = articleDf[articleDf.description.str.contains('The "Fast Money" traders share their first moves for the market open.', case=False) == False]
    articleDf = articleDf[articleDf.description.str.contains('stuff we think you', case=False) == False]

    #remove transcripts
    articleDf = articleDf[articleDf.title.str.contains('transcript', case=False) == False]

    #remove cramer
    articleDf = articleDf[articleDf.title.str.contains('cramer', case=False) == False]

    #remove articles with less than words which is the lower end of the boxplot
    articleDf = articleDf[articleDf['content'].str.split().str.len() > 300]

    #remove duplicates
    # by self-identified repeat
    articleDf = articleDf[articleDf.title.str.contains('rpt', case=False) == False]
    # by title
    articleDf = articleDf.drop_duplicates(subset=['title'], keep='first')
    # by content
    articleDf = articleDf.drop_duplicates(subset=['content'], keep='first')
    # by decription
    articleDf = articleDf.drop_duplicates(subset=['description'], keep='first')

    articleDf = articleDf.reset_index(drop=True)

    # CLEAN ORIGINAL CONTENT
    articleDf['origContent'] = articleDf['content'] 

    #Remove nonsense sentence from original content pull
    for i in articleDf.index:
        article = articleDf['origContent'].iloc[i].split('\r\n')

        # remove lines with no period
        article[:] = [sentence for sentence in article if '.' in sentence]
        # remove lines with less than 5 words
        article[:] = [sentence for sentence in article if len(sentence.split())>5]
        # remove photo credits
        article[:] = [sentence for sentence in article if not('Photo' in sentence)]
        blackList = ['get breaking news','click here','write to','subscribe','read more','read or share'
                     ,'reporting by','twitter, instagram','comment','copyright','©', 'fox', 'you', 'sign up', 'your inbox']
        # remove lines with terms that are associated with useless sentences
        article[:] = [sentence for sentence in article if not any(term in sentence.lower() for term in blackList)]
        try:
            article[0] = '<p>'+article[0]
            article[len(article)-1] = article[len(article)-1]+'</p>'
        except:
            continue
        
        articleDf.at[i,'origContent']='</p> <p>'.join(article)

    #Remove videos from cnbc links
    pat_cnbcVid = re.compile('div &gt; div\.group &gt; p:first-child"&gt;')
    articleDf['origContent'] = list(map(lambda x: pat_cnbcVid.sub('', x), articleDf['origContent']))
    pat_vid = re.compile('gt;')
    articleDf['origContent'] = list(map(lambda x: pat_vid.sub('', x), articleDf['origContent']))
    #Remove amp;
    pat_amp = re.compile('amp;')
    articleDf['origContent'] = list(map(lambda x: pat_amp.sub('', x), articleDf['origContent']))    
    
    # CLEAN CONTENT FOR FEATURE SELECTION articleDf['content'] AND CONTEXT EXTRACTION articleDf['contentWithStops'] 

    #Remove html tags
    pat_htmlTags = re.compile(r'<.*?>')
    articleDf['content'] = list(map(lambda x: pat_htmlTags.sub('', x), articleDf['origContent']))
    
    #Remove time
    pat_time = re.compile('[0-9]{0,2}:?[0-9]{1,2}\s?[aApP]\.?[mM]\.?')
    articleDf['content'] = list(map(lambda x: pat_time.sub(' ', x), articleDf['content']))

    #Remove urls
    pat_url = re.compile('[a-z]+?[.]?[a-z]+?[.]?[a-z]+[.]?[\/\/]\S+')
    articleDf['content'] = list(map(lambda x: pat_url.sub('URL', x), articleDf['content']))
    pat_https = re.compile('https://')
    articleDf['content'] = list(map(lambda x: pat_https.sub('', x), articleDf['content']))
    pat_http = re.compile('http://')
    articleDf['content'] = list(map(lambda x: pat_http.sub('', x), articleDf['content']))
    
    #Remove characters that don't separate a sentence or aren't $ signs
    # FOR context extraction
    # Remove non-ascii chars -- these get cleaned up in 'content' when we remove punctuation
    pat_nonascii = re.compile('[^\x00-\x7f]')
    articleDf['contentWithStops'] = list(map(lambda x: pat_nonascii.sub(' ', x), articleDf['content']))
    #pat_nonStops = re.compile('[^\.\?!,;\$0-9a-zA-Z]+')
    #articleDf['contentWithStops'] = list(map(lambda x: pat_nonStops.sub(' ', x), articleDf['content']))
    
    #Remove stopwords & apply lowercasing
    stopwords = [
        # dates/times
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "jan", "feb","mar", "apr", "jun", "jul", "aug", "oct", "nov", "dec", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning", "evening","today","pm","am","daily", 
        # specific article terms that are useless
        "read", "share", "file", "'s","'t", "photo", "inc", "corp", "group", "inc", "corp", "source", "bloomberg", "cnbc","cnbcs", "cnn", "reuters","bbc", "published", "broadcast","msnbc","ap","said","nbcuniversal","newsletterupgrade","nbc", "news",
        # other useless terms
        "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "co", "inc", "com", "theyve", "theyre", "theres", "heres", "didnt", "wouldn", "couldn", "didn","according", "just", "us", "ll", "times","yes","such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "don", "now", "will", "wasn", "etc", "but", "hello", "welcome", "re","my","wasnt","also","us","the", "a", "of", "have", "has", "had", "having", "hello", "welcome", "yeah", "wasn", "today", "etc", "ext","definitely", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "while", "of", "at", "by", "for", "about", "into", "through", "during", "before", "after", "to", "from", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "don", "now", "will"
    ]
    pat_stopwords = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
    articleDf['content'] = list(map(lambda x: pat_stopwords.sub(' ', x), articleDf['content'].str.lower()))

    #Remove single character words
    pat_charLim = re.compile('\s[a-zA-Z]\s')
    articleDf['content'] = list(map(lambda x: pat_charLim.sub(' ', x), articleDf['content']))

    #Remove punctuation 
    # FOR feature selection/encoding
    pat_punctuation = re.compile('[^a-zA-Z]+')
    articleDf['content'] = list(map(lambda x: pat_punctuation.sub(' ', x), articleDf['content']))

    #Remove single characters
    articleDf['content'] = list(map(lambda x: pat_charLim.sub(' ', x), articleDf['content']))

    return articleDf