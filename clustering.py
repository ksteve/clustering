# import packages
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt

the script runs. This will help us track the data.
# 
# Here is the complete script:
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from tqdm import tqdm


def category(source, m):
    try:
        return m[source]
    except:
        return 'NC'

def cleanData(path):
    data = pd.read_csv(path)
    data = data.drop_duplicates('url')
    data.to_csv(path, index=False)


def getDailyNews():

    sources = getSources()
    key = 'ef9327ef4e554ab3904bb5341d9aeb3b'
    url = 'https://newsapi.org/v1/articles?source={0}&sortBy={1}&apiKey={2}'
    responses = []
    for i, source in tqdm(enumerate(sources)):
        try:
            u = url.format(source, 'top',key)
            response = requests.get(u)
            r = response.json()
            for article in r['articles']:
                article['source'] = source
            responses.append(r)
        except:
            u = url.format(source, 'latest', key)
            response = requests.get(u)
            r = response.json()
            for article in r['articles']:
                article['source'] = source
            responses.append(r)
      
    news = pd.DataFrame(reduce(lambda x,y: x+y ,map(lambda r: r['articles'], responses)))
    news = news.dropna()
    news = news.drop_duplicates()
    d = mapping()
    news['category'] = news['source'].map(lambda s: category(s, d))
    news['scraping_date'] = datetime.now()

    try:
        aux = pd.read_csv('/home/news/news.csv')
    except:
        aux = pd.DataFrame(columns=list(news.columns))
        aux.to_csv('/home/news/news.csv', encoding='utf-8', index=False)

    with open('/home/news/news.csv', 'a') as f:
        news.to_csv(f, header=False, encoding='utf-8', index=False)

    cleanData('/home/news/news.csv')
    
    print('Done')


if __name__ == '__main__':
    getDailyNews()
# Ok, now this script needs to run repetitively to collect the data. 
# 
# To do this:
# 
# I uploaded the script to my linux server at this path /home/news/news.py then I created a crontab schedule to tell my server to run news.py every 5 minutes. To do this: 
#  - from the terminal, type crontab -e to edit the crontab file
#  - add this line to the end of the file using nano or vim: \*/5 \* \* \* \* /root/anaconda2/bin/python /home/news/news.py
#     (put absolute paths for your executables).
#     
#   basically what this command tells the server is: "for every 5 minutes (\*/5) of every hour (\*) of every day of the month (\*) of every month (\*) and whatever the day of the week (\*), run the news.py script.
#     
#  -  give your script the execution permission. Otherwise, this won't work: chmod +x news.py 
# 

# Now that the data has been collected, we will start anlayzing it :
# 
# - We'll have a look at the dataset and inspect it
# - We'll apply some preoprocessings on the texts: tokenization, tf-idf
# - We'll cluster the articles using two different algorithms (Kmeans and LDA)
# - We'll visualize the clusters using Bokeh and pyldavis

# ## 3 - Data analysis

# ### 3 - 1 - Data discovery


# pandas for data manipulation
import pandas as pd
pd.options.mode.chained_assignment = None
# nltk for nlp
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# list of stopwords like articles, preposition
stop = set(stopwords.words('english'))
from string import punctuation
from collections import Counter
import re
import numpy as np



# Many mixed topics are included in the "general" category.
# 
# This gives us a very superficial classificaion of the news. It doesn't tell us the underlying topics, nor the keywords and and the most relevant news per each category. 
# 
# To get that sort of information, we'll have to process the descriptions of each article since these variables naturally carry more meanings.
# 
# Before doing that, let's focus on the news articles whose description length is higher than 140 characters (a tweet length). Shorter descriptions happen to introduce lots of noise.

# In[14]:


# remove duplicate description columns
data = data.drop_duplicates('description')


# In[15]:


# remove rows with empty descriptions
data = data[~data['description'].isnull()]


# In[16]:


data.shape


# In[17]:


data['len'] = data['description'].map(len)


# In[18]:


data = data[data.len > 140]
data.reset_index(inplace=True)
data.drop('index', inplace=True, axis=1)


# In[19]:


data.shape


# We are left with 30% of the initial dataset.

# ### 3 - 2 - Text processing : tokenization

# Now we start by building a tokenizer. This will, for every description:
# 
# - break the descriptions into sentences and then break the sentences into tokens
# - remove punctuation and stop words
# - lowercase the tokens

# In[20]:


def tokenizer(text):
    try:
        tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]
        
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent

        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        tokens = list(filter(lambda t: t not in punctuation, tokens))
        tokens = list(filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``', 
                                            u'\u2014', u'\u2026', u'\u2013'], tokens))
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))

        return filtered_tokens
    except Error as e:
        print(e)


# A new column 'tokens' can be easily created using the map method applied to the 'description' column.

# In[21]:


data['tokens'] = data['description'].map(tokenizer)


# The tokenizer has been applied to each description through all rows. Each resulting value is then put into the 'tokens' column that is created after the assignment. Let's check what the tokenization looks like for the first 5 descriptions:

# In[22]:


for descripition, tokens in zip(data['description'].head(5), data['tokens'].head(5)):
    print('description:', descripition)
    print('tokens:', tokens)
    print() 


# Let's group the tokens by category, apply a word count and display the top 10 most frequent tokens. 

# In[23]:


def keywords(category):
    tokens = data[data['category'] == category]['tokens']
    alltokens = []
    for token_list in tokens:
        alltokens += token_list
    counter = Counter(alltokens)
    return counter.most_common(10)


# In[24]:


for category in set(data['category']):
    print('category :', category)
    print('top 10 keywords:', keywords(category))
    print('---')


# Looking at these lists, we can formulate some hypotheses:
# 
# - the sport category deals with the champions' league, the footbal season and NFL
# - some tech articles refer to Google
# - the business news seem to be highly correlated with US politics and Donald Trump (this mainly originates from us press)
# 
# Extracting the top 10 most frequent words per each category is straightforward and can point to important keywords. 
# 
# However, although we did preprocess the descriptions and remove the stop words before, we still end up with words that are very generic (e.g: today, world, year, first) and don't carry a specific meaning that may describe a topic.
# 
# As a first approach to prevent this, we'll use tf-idf

# ### 3 - 3 - Text processing : tf-idf

# tf-idf stands for term frequencey-inverse document frequency. It's a numerical statistic intended to reflect how important a word is to a document or a corpus (i.e a collection of documents). 
# 
# To relate to this post, words correpond to tokens and documents correpond to descriptions. A corpus is therefore a collection of descriptions.
# 
# The tf-idf a of a term t in a document d is proportional to the number of times the word t appears in the document d but is also offset by the frequency of the term t in the collection of the documents of the corpus. This helps adjusting the fact that some words appear more frequently in general and don't especially carry a meaning.
# 
# tf-idf acts therefore as a weighting scheme to extract relevant words in a document.
# 
# $$tfidf(t,d) = tf(t,d) . idf(t) $$
# 
# $tf(t,d)$ is the term frequency of t in the document d (i.e. how many times the token t appears in the description d)
# 
# $idf(t)$ is the inverse document frequency of the term t. it's computed by this formula:
# 
# $$idf(t) = log(1 + \frac{1 + n_d}{1 + df(d,t)}) $$
# 
# - $n_d$ is the number of documents
# - $df(d,t)$ is the number of documents (or descriptions) containing the term t
# 
# Computing the tfidf matrix is done using the TfidfVectorizer method from scikit-learn. Let's see how to do this:

# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer

# min_df is minimum number of documents that contain a term t
# max_features is maximum number of unique tokens (across documents) that we'd consider
# TfidfVectorizer preprocesses the descriptions using the tokenizer we defined above

vectorizer = TfidfVectorizer(min_df=10, max_features=10000, tokenizer=tokenizer, ngram_range=(1, 2))
vz = vectorizer.fit_transform(list(data['description']))


# In[26]:


vz.shape


# vz is a tfidf matrix. 
# 
# - its number of rows is the total number of documents (descriptions) 
# - its number of columns is the total number of unique terms (tokens) across the documents (descriptions)
# 
# $x_{dt}  = tfidf(t,d)$ where $x_{dt}$ is the element at the index (d,t) in the matrix.

# Let's create a dictionary mapping the tokens to their tfidf values 

# In[27]:


tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
tfidf.columns = ['tfidf']


# We can visualize the distribution of the tfidf scores through an histogram

# In[28]:


tfidf.tfidf.hist(bins=50, figsize=(15,7))


# Let's display the 30 tokens that have the lowest tfidf scores 

# In[29]:


tfidf.sort_values(by=['tfidf'], ascending=True).head(30)


# Not surprisingly, we end up with a list of very generic words. These are very common across many descriptions. tfidf attributes a low score to them as a penalty for not being relevant. Words likes may, one, new, back, etc.
# 
# You may also notice that Trump, Donald, U.S and president are part of this list for being mentioned in many articles. So maybe this may be the limitation of the algorithm.
# 
# Now let's check out the 30 words with the highest tfidf scores.

# In[30]:


tfidf.sort_values(by=['tfidf'], ascending=False).head(30)


# We end up with less common words. These words naturally carry more meaning for the given description and may outline the underlying topic.

# As you've noticed, the documents have more than 4000 features (see the vz shape). put differently, each document has more than 4000 dimensions.
# 
# If we want to plot them like we usually do with geometric objects, we need to reduce their dimension to 2 or 3 depending on whether we want to display on a 2D plane or on a 3D space. This is what we call dimensionality reduction.
# 
# To perform this task, we'll be using a combination of two popular techniques: Singular Value Decomposition (SVD) to reduce the dimension to 50 and then t-SNE to reduce the dimension from 50 to 2. t-SNE is more suitable for dimensionality reduction to 2 or 3.
# 
# Let's start reducing the dimension of each vector to 50 by SVD.

# In[31]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=0)
svd_tfidf = svd.fit_transform(vz)


# In[32]:


svd_tfidf.shape


# Bingo. Now let's do better. From 50 to 2!

# In[33]:


from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)


# Let's check the size.

# In[34]:


tsne_tfidf.shape


# Each description is now modeled by a two dimensional vector. 
# 
# Let's see what tsne_idf looks like.

# In[35]:


tsne_tfidf


# We're having two float numbers per discription. This is not interpretable at first sight. 
# 
# What we need to do is find a way to display these points on a plot and also attribute the corresponding description to each point.
# 
# matplotlib is a very good python visualization libaray. However, we cannot easily use it to display our data since we need interactivity on the objects. One other solution could be d3.js that provides huge capabilities in this field. 
# 
# Right now I'm choosing to stick to python so I found a tradeoff : it's called Bokeh.
# 
# "Bokeh is a Python interactive visualization library that targets modern web browsers for presentation. Its goal is to provide elegant, concise construction of novel graphics in the style of D3.js, and to extend this capability with high-performance interactivity over very large or streaming datasets. Bokeh can help anyone who would like to quickly and easily create interactive plots, dashboards, and data applications." To know more, please refer to this <a href="http://bokeh.pydata.org/en/latest/"> link </a>

# Let's start by importing bokeh packages and initializing the plot figure.

# In[36]:


import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook


# In[37]:


output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="tf-idf clustering of the news",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)


# In[38]:


tfidf_df = pd.DataFrame(tsne_tfidf, columns=['x', 'y'])
tfidf_df['description'] = data['description']
tfidf_df['category'] = data['category']


# Bokeh need a pandas dataframe to be passed as a source data. this is a very elegant way to read data.

# In[39]:


plot_tfidf.scatter(x='x', y='y', source=tfidf_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"description": "@description", "category":"@category"}
show(plot_tfidf)


# Bokeh charts offer many functionalities:
# 
# - navigating in the data
# - zooming
# - hovering on each data point and displaying the corresponding description
# - saving the chart
# 
# When the description popup doesn't show properly you have to move the data point slightly on the left.
# 
# By hovering on each news cluster, we can see groups of descriptions of similar keywords and thus referring to the same topic.
# 
# Now we're going to use clustering algorithms on the tf-idf matrix.

# ## 4 - Clustering 
# ### 4 - 1 - KMeans

# Our starting point is the tf-idf matrix vz. Let's check its size again.

# In[40]:


vz.shape


# This matrix can be seen as a collection of (x) high-dimensional vectors (y). Some algorithms like K-means can crunch this data structure and produce blocks of similar or "close" data points based on some similarity measure like the euclidean distance.
# 
# One thing to know about Kmeans is that it needs the number of clusters up front. This number is usually found by trying different values until the result looks satisfactory.
# 
# I found that 20 was a good number that separates the dataset nicely.

# In[41]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.cluster import MiniBatchKMeans

num_clusters = 30
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
kmeans = kmeans_model.fit(vz)
kmeans_clusters = kmeans.predict(vz)
kmeans_distances = kmeans.transform(vz)


# Let's see the five first description and the associated cluster

# In[42]:


for (i, desc),category in zip(enumerate(data.description),data['category']):
    if(i < 5):
        print("Cluster " + str(kmeans_clusters[i]) + ": " + desc + 
              "(distance: " + str(kmeans_distances[i][kmeans_clusters[i]]) + ")")
        print('category: ',category)
        print('---')


# This doesn't tell us much. What we need to look up are the "hot" keywords that describe each clusters. 

# In[43]:


sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %d:" % i)
    aux = ''
    for j in sorted_centroids[i, :10]:
        aux += terms[j] + ' | '
    print(aux)
    print() 


# Looking at these clusters you can roughly have an idea of what's going on.
# 
# 
# Let's plot these clusters. To do this we need to reduce the dimensionality of kmeans_distances to 2.

# In[44]:


tsne_kmeans = tsne_model.fit_transform(kmeans_distances)


# Let's use a color palette to assign different colors to each cluster 

# In[45]:


import numpy as np

colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
"#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
"#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
"#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])

plot_kmeans = bp.figure(plot_width=700, plot_height=600, title="KMeans clustering of the news",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)


# In[46]:


kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
kmeans_df['cluster'] = kmeans_clusters
kmeans_df['description'] = data['description']
kmeans_df['category'] = data['category']


# In[47]:


plot_kmeans.scatter(x='x', y='y', 
                    color=colormap[kmeans_clusters], 
                    source=kmeans_df)
hover = plot_kmeans.select(dict(type=HoverTool))
hover.tooltips={"description": "@description", "category": "@category", "cluster":"@cluster"}
show(plot_kmeans)


# It looks like clusters are separated nicely. By hovering on each one of them you can see the corresponding descriptions. At first sight you could notice that they deal approximately with the same topic. This is coherent since we build our clusters using similarities between relevant keywords.
# 
# We can also notice that within the same cluster, many subclusters are isolated from one another. This gives an idea about the global topic as well as the 
# 
# 
# Kmeans separates the documents into disjoint clusters. the assumption is that each cluster is attributed a single topic.
# 
# However, descriptions may in reality be characterized by a "mixture" of topics. We'll cover how to deal with this problem with the LDA algorithm.

# ### 4 - 2 - 1 - Latent Dirichlet Allocation (with Bokeh)

# Let's apply LDA on the data set. We'll set the number of topics to 20.

# In[48]:


import lda
from sklearn.feature_extraction.text import CountVectorizer


# In[49]:


import logging
logging.getLogger("lda").setLevel(logging.WARNING)


# In[50]:


cvectorizer = CountVectorizer(min_df=4, max_features=10000, tokenizer=tokenizer, ngram_range=(1,2))
cvz = cvectorizer.fit_transform(data['description'])

n_topics = 20
n_iter = 2000
lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
X_topics = lda_model.fit_transform(cvz)


# In[51]:


n_top_words = 8
topic_summaries = []

topic_word = lda_model.topic_word_  # get the topic words
vocab = cvectorizer.get_feature_names()
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# In[52]:


tsne_lda = tsne_model.fit_transform(X_topics)


# In[53]:


doc_topic = lda_model.doc_topic_
lda_keys = []
for i, tweet in enumerate(data['description']):
    lda_keys += [doc_topic[i].argmax()]


# In[54]:


plot_lda = bp.figure(plot_width=700, plot_height=600, title="LDA topic visualization",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)


# In[55]:


lda_df = pd.DataFrame(tsne_lda, columns=['x','y'])
lda_df['description'] = data['description']
lda_df['category'] = data['category']


# In[56]:


lda_df['topic'] = lda_keys
lda_df['topic'] = lda_df['topic'].map(int)


# In[57]:


plot_lda.scatter(source=lda_df, x='x', y='y', color=colormap[lda_keys])

hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips={"description":"@description", "topic":"@topic", "category":"@category"}
show(plot_lda)


# Better separation between the topics.
# 
# No more dominant topic.
# 

# ### 4 - 2 - 2 Visualization of the topics using pyLDAvis
# 
# 
# Now we're going to use a more convenient visualization to explore LDA topics. It's called pyldavis.

# In[58]:


lda_df['len_docs'] = data['tokens'].map(len)


# In[59]:


def prepareLDAData():
    data = {
        'vocab': vocab,
        'doc_topic_dists': lda_model.doc_topic_,
        'doc_lengths': list(lda_df['len_docs']),
        'term_frequency':cvectorizer.vocabulary_,
        'topic_term_dists': lda_model.components_
    } 
    return data


# In[60]:


ldadata = prepareLDAData()


# In[61]:


import pyLDAvis


# In[62]:


pyLDAvis.enable_notebook()


# In[63]:


prepared_data = pyLDAvis.prepare(**ldadata)


# In[64]:


pyLDAvis.save_html(prepared_data,'./pyldadavis.html')


# ## 5 - Conclusion
# 
# In this post we explored many topics. 
# 
# - We set up a script to automatically extract newsfeed data from a REST API called newsapi.
# - We processed the raw text by using different tools (pandas, nltk, scikit-learn)
# - We applied tf-idf statistics as a natural language preprocessing technique
# - We created clusters on top of the tf-idf matrix using the KMeans algorithm and visualized them using Bokeh
# - We extracted topics using the Latent Dirichlet Allocation algorithm and visualized them using Bokeh and pyldavis
# 
# Different techniques have been used but I'm pretty sure there's plenty of better methods. In fact, one way to extend this tutorial could be to dive in: 
# 
# - word2vec and doc2vec to model the topics
# - setting up a robust way to select the number of clusters/topics up front
# 
# Thanks for reading ! Don't hesitate to comment if you have a suggestion or an improvement. 

# ## 6 - References
# 
# - https://newsapi.org/
# - http://scikit-learn.org/stable/modules/feature_extraction.html
# - https://en.wikipedia.org/wiki/Tf%E2%80%93idf
# - http://pythonhosted.org/lda/
# - http://nbviewer.jupyter.org/github/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb#topic=3&lambda=0.87&term=
