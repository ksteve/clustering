import lda
import json
import logging
import sys
import nltk
import numpy as np
import os.path

from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN
from sklearn.metrics import adjusted_rand_score, jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.externals import joblib
from nltk.stem.porter import PorterStemmer
from newspaper import Article as Extractor
from optparse import OptionParser
from time import time
import itertools

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def jaccard(vector):
    #
    sims = cosine_similarity(vector)
    return sims

#init

timelines = []
stemmer = PorterStemmer()
newsapi = NewsApiClient(api_key='1f62f144d9584aaeb3fb553f42c989a6')

if os.path.exists('./timelines.pkl'):
    joblib.load('./timelines.pkl', timelines)

#start

top_headlines = newsapi.get_top_headlines(language='en')

#get urls from news api
urls = [*filter(None,[*map(lambda x: x['url'], top_headlines['articles'])])]

# extract from article urls
articles = []
for u in urls:
    try:
        content = Extractor(u)
        content.download()
        content.parse()
        content.nlp()
        article = {}
        article['title'] = content.title
        article['url'] = u
        stemmed_words = set(stem_tokens(content.keywords, stemmer))
        article['keywords'] = ' '.join(sorted(stemmed_words))
        articles.append(article)
    except:
        continue

#add existing articles to new articles
if len(timelines) > 0:
    recent_timelines = [*map(lambda t: t[0], timelines)]
    articles = recent_timelines + articles

#tf-idf the articles
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform([*map(lambda x: x['keywords'],articles)])

#jaccard similarity on vector
sims = jaccard(X)

#
new_timelines = set()
test = {}
for i, y in enumerate(sims):
    tm = []
    #print(articles[i]['title'])
    #print()
    #print()
    for idx, x in enumerate(y):
        if idx != i and x >= 0.1:
            tm.append(idx)
            #print(articles[idx]['title'])
    #print(tm)
    test[i] = tm

    #new_timelines.add(frozenset(tm))
    #print('---------------------------------')
    #print()
    #print()
print(test)
print()

#print(new_timelines)


# #joblib.dump(model, 'clusters.pkl')


