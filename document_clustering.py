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
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.externals import joblib
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from nltk.stem.porter import PorterStemmer
from goose3 import Goose
from newspaper import Article as Extractor
from optparse import OptionParser
from time import time
import itertools
import string
from gensim.models import HdpModel
from gensim.sklearn_api.hdp import HdpTransformer
import gensim

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

def jaccard(vector):
    #
    sims = cosine_similarity(vector)
    return sims

#init
#opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
g = Goose()
timelines = []
stemmer = PorterStemmer()
newsapi = NewsApiClient(api_key='1f62f144d9584aaeb3fb553f42c989a6')
translator = str.maketrans('', '', string.punctuation)

if os.path.exists('./timelines.pkl'):
    joblib.load('./timelines.pkl', timelines)

#start
#,abc-news,buzzfeed,daily-mail,bbc-news,cbc-news,cnn
top_headlines = newsapi.get_top_headlines(language='en', sources='google-news', page_size=100)

#get urls from news api
urls = [*filter(None,[*map(lambda x: x['url'], top_headlines['articles'])])]

# extract from article urls
articles = []
for u in urls:
    try:
      #  response = opener.open(u)
      #  raw_html = response.read()
        #content = Extractor(u)
        content = g.extract(url=u)
        #text = content.cleaned_text.lower()
        text = content.cleaned_text.replace("\n", " ")
        #no_punc = text.translate(translator)

        #content.download()
        #content.parse()
        #content.nlp()
        article = {}

       # stemmed_words = set(stem_tokens(content.cleaned_text, stemmer))
        article['keywords'] = text
        article['url'] = u
        article['title'] = content.title
        articles.append(article)
    except:
        continue

for a in articles:
    print(a['url'])

#add existing articles to new articles
if len(timelines) > 0:
    recent_timelines = [*map(lambda t: t[0], timelines)]
    articles = recent_timelines + articles


#tf-idf the articles
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
X = vectorizer.fit_transform([*map(lambda x: x['keywords'],articles)])
#print(vectorizer.get_feature_names())

svd = TruncatedSVD(n_components=100, n_iter=100)
lda = LatentDirichletAllocation(n_components=len(articles))
L = lda.fit(X)
X = svd.fit(X)
#normalizer = Normalizer(copy=False)
#lsa = make_pipeline(svd, normalizer)
#X = lsa.fit_transform(X)

terms = vectorizer.get_feature_names()
for i, comp in enumerate(L.components_):
    termsInComp = zip(terms,comp)
    sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:20]
    print("Concept %d:" % i)
    for term in sortedTerms:
        print(term[0])
    print(" ")


#jaccard similarity on vector
sims = jaccard(X)
print(sims)
#
new_timelines = set()
test = []
for i, y in enumerate(sims):
    tm = []
    #print(articles[i]['title'])
    #print()
    #print()
    for idx, x in enumerate(y):
        if x >= 0.1:
            tm.append(idx)
            #print(articles[idx]['title'])
    #print(tm)
    if not sorted(tm) in test:
        test.append(sorted(tm))

    #new_timelines.add(frozenset(tm))
    #print('---------------------------------')
    #print()
    #print()
print(test)
print()

#print(new_timelines)


# #joblib.dump(model, 'clusters.pkl')


