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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from goose3 import Goose
from time import time
import string
from gensim.models import HdpModel
from gensim.sklearn_api.hdp import HdpTransformer
from gensim import corpora, models
import gensim

stopset = list(set(stopwords.words('english'))) + list(string.punctuation)

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if not w in stopset]
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
        text = tokenize(text)

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


#add existing articles to new articles
if len(timelines) > 0:
    recent_timelines = [*map(lambda t: t[0], timelines)]
    articles = recent_timelines + articles

texts = [*map(lambda x: x['keywords'],articles)]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

hdp = HdpModel(corpus, dictionary)
#print(hdp.print_topics(num_topics=3, num_words=10))
print(hdp.show_topics(num_topics=-1, num_words=10))

topics = hdp.print_topics(num_topics=-1)
texst = 1
