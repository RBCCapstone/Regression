import re

def highlight(phrase, text):
    findit = re.finditer(phrase,text, re.IGNORECASE)
    matches = list(findit)
    for i in range(len(matches)-1,-1,-1):
        match = matches[i]
        start = match.span()[0]
        end = match.span()[1]
        text = text[:start]+'<strong>'+phrase+'</strong>'+text[end:]
        
    #start = re.search(phrase,text, re.IGNORECASE).start()
    #end = re.search(phrase, text, re.IGNORECASE).end()
    #bold = text[:start]+'<strong>'+phrase.upper()+'</strong>'+text[end:]
    return text

def articletags(tags,article):
    for tag in tags:
        article = highlight(tag,article)
    return article
        
