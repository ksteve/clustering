# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

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
from newspaper import Article
from optparse import OptionParser
from time import time
import itertools

stemmer = PorterStemmer()


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
    print(sims)
    for i, x in enumerate(sims[0]):
        if x < 1.0 and x > 0.0:
            print(articles[i]['title'])
  



newsapi = NewsApiClient(api_key='1f62f144d9584aaeb3fb553f42c989a6')
top_headlines = newsapi.get_top_headlines(language='en', q="Trump")

#print(top_headlines)
headlines = top_headlines['articles']
urls = [*map(lambda x: x['url'], headlines)]
urls = [*filter(None, urls)][:10]



articles = []
for u in urls:
    try:
        article = Article(u)
        article.download()
        article.parse()
        article.nlp()
        blah = {}
        blah['title'] = article.title
        stemmed_words = set(stem_tokens(article.keywords, stemmer))
        blah['keywords'] = ' '.join(sorted(stemmed_words))
        articles.append(blah)
    except:
        continue



vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform([*map(lambda x: x['keywords'],articles)])
jaccard(X)

#cvectorizer = CountVectorizer(min_df=0.1, stop_words='english')
#cvz = cvectorizer.fit_transform(articles)

svd = TruncatedSVD(random_state=0)
svd_tfidf = svd.fit_transform(X)

#tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
#tsne_tfidf = tsne_model.fit_transform(svd_tfidf)

#if os.path.exists('./clusters.pkl'):
#    model = joblib.load('clusters.pkl')
#    model.fit_predict(X)
#else:
true_k = 3
model = KMeans(n_clusters=true_k)
#model = DBSCAN(metric='cosine')
#model = AffinityPropagation(preference=-10)
model.fit(X)

clusters = model.labels_.tolist()
#joblib.dump(model, 'clusters.pkl')

''' cvectorizer = CountVectorizer(min_df=0.1, max_features=10000, ngram_range=(1,2), stop_words='english')
cvz = cvectorizer.fit_transform(descriptions)

n_topics = 20
n_iter = 2000
lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
X_topics = lda_model.fit_transform(cvz)

n_top_words = 8
topic_summaries = []

topic_word = lda_model.topic_word_  # get the topic words
vocab = cvectorizer.get_feature_names()
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

tsne_lda = tsne_model.fit_transform(X_topics) '''

print("Top terms per cluster:")
#order_centroids = af.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(model.n_clusters):
    print("Cluster %d:" % (i+1))
    for z in range(len(clusters)):    
        if clusters[z] == i:
            print(articles[z]['title'])
            print()
            print()
    print()
 #   for ind in order_centroids[i, :10]:
#        print(' %s' % terms[ind])

print("\n")
print("Prediction")

# Y = vectorizer.transform(["chrome browser to open."])
# prediction = model.predict(Y)
# print(prediction)

# Y = vectorizer.transform(["My cat is hungry."])
# prediction = model.predict(Y)
# print(prediction)

