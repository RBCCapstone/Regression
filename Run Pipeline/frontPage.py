import os
from pathlib import Path
import json
import pandas as pd
import re
#import pipeline as pl


# In[1]:
def highlight(phrase, text):
    findit = re.finditer(phrase,text, re.IGNORECASE)
    matches = list(findit)
    for i in range(len(matches)-1,-1,-1):
        match = matches[i]
        start = match.span()[0]
        end = match.span()[1]
        text = text[:start]+'<span class=\"highlight\">'+match.group()+'</span>'+text[end:]
        
    #start = re.search(phrase,text, re.IGNORECASE).start()
    #end = re.search(phrase, text, re.IGNORECASE).end()
    #bold = text[:start]+'<strong>'+phrase.upper()+'</strong>'+text[end:]
    return text

def highlightarticle(tags,article):
    for tag in tags:
        article = highlight(tag,article)
    return article
        

def FrontPage(articleDB, trendingTermsDB):
    # number of top articles
    # todo; change to only 'predicted relevant' articles
    numArts = 15
    
    # get articles
    art = articleDB.iloc[0:numArts][['title','source', 'date', 'origContent', 'url']]
    
    art['tags'] = list(map(lambda x: x.split(','), articleDB.iloc[0:numArts]['tags_top_5']))
    for i in art.index:
        art['origContent'] = highlightarticle(art['tags'].iloc[i],art['origContent'].iloc[i])
    
    
    # grab related article IDs
    rel_arts = list(map(lambda x: x.split(','), articleDB.iloc[0:numArts]['related_articles']))
    # use IDs to grab related article title, source, url, turn into little dictionaries and add to art
    art['related_articles'] = list(map(lambda num: articleDB.iloc[num][['title','source','url']].to_dict(orient='records'), rel_arts))
    art = art.sort_values(by=['date'], axis = 0, ascending = False)    
    artDict = art.to_dict(orient='records')
    
    
    # get top terms
    tuples = [tuple(x) for x in trendingTermsDB.values]
    topTerms = tuples[:15]
    
    # output final json
    frontpage = {"topterms":topTerms, "articles":artDict}
    with open("data.json", "w") as write_file:
        json.dump(frontpage, write_file)
    
    return frontpage