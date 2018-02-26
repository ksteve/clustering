from newsapi import NewsApiClient
from goose3 import Goose
from rake_nltk import Rake
import json
import time
import timeout_decorator
import sqlite3
import re

def create_database():
    sql = 'create table if not exists ' + 'headlines(url TEXT, title TEXT, text TEXT, source TEXT, publishDate TEXT, unique (url))'
    c.execute(sql)
    conn.commit()

def insert_list(headlines):
    for item in headlines:
        try:
            c.execute('''insert into headlines values (:url,:title,:text,:source,:publishDate)''', item)
            conn.commit()
        except:
            continue

def getContentFromHeadline(item):
        article = {}
        content = g.extract(url=item['url'])
        article['url'] = item['url']
        article['title'] = item['title']
        article['text'] = content.cleaned_text
        #new_text = re.sub('[^a-zA-Z0-9 ]+', ' ', content.cleaned_text)
        #r.extract_keywords_from_text(new_text)
        #print(r.get_ranked_phrases())
        article['source'] = item['source']['name']
        article['publishDate'] = item['publishedAt']
        return article

g = Goose()
r = Rake()
conn = sqlite3.connect('top_headlines.db')
c = conn.cursor()

create_database()

newsapi = NewsApiClient(api_key='1f62f144d9584aaeb3fb553f42c989a6')

#'abc-news,bbc-news,buzzfeed,cnn,daily-mail,fox-news,google-news,reuters,the-washington-post,the-new-york-times'
top_headlines = newsapi.get_top_headlines(language='en', sources='bbc-news', page_size=100)
#get urls
headlines = top_headlines['articles']
articles = []

for item in headlines:
    try:
        art = getContentFromHeadline(item)
        if art['text'] != '':
            articles.append(art)
    except:
        continue

insert_list(articles)
conn.close()


