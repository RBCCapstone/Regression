
# coding: utf-8

# Takes the following inputs: manual (1,0), Pull from, pull to (can be left blank), CompanyList (1-5) , where 5 is all 19 companies

#import libraries
import requests
import csv
import os
import pandas as pd
from datetime import date, timedelta


# In[1]:

# News API
# This script pulls data from various news sources

#import API key (environment variable)
#newsapiKey = os.environ['NEWSAPI_KEY']
newsapiKey = 'abd1cde781dc46b385045b20e214a7e8'

def News(querylist, sources, fromdate, todate):


    #Create a string to query the url    
    completequery = ""    
    for i in range(len(querylist)): #defaults at index 0  
        #Create a string to query the url     
        if i < len(querylist)-1:
            completequery += querylist[i]
            completequery += " OR "
        else:
            completequery += querylist[i]
    
     
    #Inform user on articles to print           
    print("Gathering articles on "+ completequery+ " from: "+fromdate+" to "+todate)
    
    #Find the first page
    main_url = " https://newsapi.org/v2/everything?q=(" +completequery+ ")&sources=" + sources + "&from=" + fromdate + "&to=" + todate + "&pageSize=100&page=1&apiKey=" + newsapiKey  

    
    # fetching data in json format
    open_bbc_page = requests.get(main_url).json() 
    totalResults = open_bbc_page["totalResults"]
    print(totalResults)
    
    #Write to dataframe by page, until all articles in URL are written
    df = pd.DataFrame()
    j = 1
    df = articlesToDF(main_url, j, df) #update df, page 1 first
    totalResults = totalResults - 100 

    
    while int(totalResults) > 0:
        j = j + 1 #start printing to csv at page 2
        main_url = " https://newsapi.org/v2/everything?q=(" + completequery + ")&sources=" + sources + "&from=" + fromdate + "&to=" + todate + "&pageSize=100&page=" + str(j) + "&apiKey=" + newsapiKey
        df = articlesToDF(main_url, j, df)
        totalResults = totalResults - 100
        
    return df

    
def articlesToDF(main_url, k, df):
    # getting all articles in a string article
    open_bbc_page = requests.get(main_url).json()  
    article = open_bbc_page["articles"]
    
    # empty list which will contain all trending news
    titles = []
    description = []
    url = []
    publishedAt = []
    content = []
    source = []
    
    for ar in article:
        titles.append(ar["title"])
        description.append(ar["description"])
        url.append(ar["url"])
        source.append(ar["source"]["id"])
        publishedAt.append(ar["publishedAt"])
        content.append(ar["content"])                
    
    # Translate Array columns into a dataframe to append to the main dataframe
    tempdf = pd.DataFrame()
    tempdf["title"]=titles
    tempdf["description"]=description
    tempdf["url"]=url
    tempdf["source"]=source
    tempdf["date"]=publishedAt
    tempdf["content"]=content
    
    if df.shape[0] == 0:
        df = tempdf
    else:
        df = df.append(tempdf)
    
    return df
    
    

# In[2]:


def sortarticles():
    #If You would like the articles sorted as well, run this code before opening the articles.csv file 
    path = 'Data'
    
    articles = pd.read_csv("articles.csv", header= None)
    articles.columns = ["index", "title", "description", "url", "source", "date", "content"]
    
    # convert column to datetype
    articles['date']=pd.to_datetime(articles.date)
    
    #Sort by date and export as xlsx (easier to work with as xlsx)
    articles = articles.sort_values(by='date')
    
    
    if not os.path.exists(path):
        os.makedirs(path)

    writer = pd.ExcelWriter(os.path.join(path, 'newsApiOutput.xlsx'), engine='xlsxwriter')
    articles.to_excel(writer,'Sheet1')
    writer.save()

# In[3]:


# Driver Code
def main(CompanyList=6, pull_from = None, pull_to = None):
    # function call
    
    #News APi can only take 20 queries, different querying alternatives are be
    #oldquerylist = ["Amazon", "Walmart", "Home Depot", "Comcast", "Disney", "Netflix", "McDonald's", "Costco", "Lowe's", "Twenty-First Century", "Century Fox", "Starbucks", "Charter Communications", "TJX", "American Tower", "Simon Property", "Las Vegas Sands", "Crown Castle", "Target", "Carnival", "Marriott", "Sherwin-Williams", "Prologis"]
    
    #AgriCompaniesStocks = ["GPRE", "CF", "SMG", "TSN", "DF", "NTR", "MOS", "ADM", "FDP", "CVGW"]
    #AgriCompanies= ["(Green Plains)", "(CF Industries)", "(Miracle-Gro)", "(Miracle Gro)", "(Tyson Foods)", "(Dean Foods)", "Nutrien", "(Mosaic Company)", "(Archer-Daniels)","Archer Daniels", "(Del Monte)", "(Calavo Growers)"]
    #SourcesPt1 = "abc-news,al-jazeera-english,associated-press,australian-financial-review,axios,bbc-news,bloomberg,business-insider,cbc-news,cbs-news,cnbc,cnn,financial-post,financial-times,fortune,fox-news,google-news,google-news-ca,independent,msnbc,national-greographic"
    #SourcesPt2 = "national-review, nbc-news,newsweek,new-york-magazine,politico,recode,reuters,new-scientist,techcrunch,the-globe-and-mail,the-economist,the-huffinton-post,the-new-york-times,the-wall-street-journal,the-washington-post,time,usa-today,wired"
    BusinessSources = "bloomberg,cnbc,fortune,financial-times,financial-post,the-economist,the-wall-street-journal, reuters, business-insider, the-globe-and-mail, the-washington-post, the-new-york-times, cnn, fox-news, associated-press, cbc-news, cnbc, msnbc, nbc-news, usa-today" #business-insider excluded. 
    
    #Define Dates to Gather Data, can set manual dates or use Today - 1
    #Today's date 
    #today = datetime.today().strftime('%Y-%m-%dT%H:%M:00')
    #yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:00')
    
    if not pull_from:
        pull_to = str(date.today())
        pull_from = str(date.today() - timedelta(days=6))

    #Define Companies to query on, if more than one word, include brackets
    RetailCompaniesStocks = ["GPS", "FL", "LB", "MAC", "KIM", "TJX", "CVS", "HD", "BBY", "LOW"]
    RetailCompanies1 = ["(Gap Inc)", "(Foot Locker)", "(L Brands)", "Macerich", "Kimco", "TJX", "CVS", "(Home Depot)", "(Best Buy)", "(Lowe's)" ]
    RetailCompanies2 = ["Walmart"]
    RetailCompanies3 = ["(Target's)", "TGT"]
    RetailCompanies4 = ["Amazon"]
    RetailCompanies5 = ["Walgreens", "Kohl's", "(Dollar General)", "(Bed Bath and Beyond)", "Safeway","Kroger"]
    RetailCompaniesAll = ["(Gap Inc)", "(Foot Locker)", "(L Brands)", "Macerich", "Kimco", "TJX", "CVS", "(Home Depot)", "(Best Buy)", "(Lowe's)","Walmart", "(Target's)", "TGT", "Amazon", "Kroger","Walgreens", "Kohl's", "(Dollar General)", "(Bed Bath and Beyond)", "Safeway" ]
    
    
    
    #Run to collect articles that fit within your query (for Team use)
    if CompanyList == 6:
        df = News(RetailCompaniesAll, BusinessSources, pull_from, pull_to)
    elif CompanyList == 1:
        News(RetailCompanies1, BusinessSources, pull_from, pull_to)
    elif CompanyList == 2:
        News(RetailCompanies2, BusinessSources, pull_from, pull_to)
    elif CompanyList == 3:
        News(RetailCompanies3, BusinessSources, pull_from, pull_to)
    elif CompanyList == 4:
        News(RetailCompanies4, BusinessSources, pull_from, pull_to)
    else:
        News(RetailCompanies5, BusinessSources, pull_From, pull_To)


    #Function call below to sort the articles by date
    df.sort_values(by="date")
    
    return df
    
