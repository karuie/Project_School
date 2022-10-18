# scraped from the Arxiv website
# then used the data mining tools to analyse the scraped data

import pandas as pd
import numpy as np
from sklearn import preprocessing

# Translate the jsonlines format to python DataFrame
jsonObj = pd.read_json(path_or_buf="/Users/yimin/project/dataMining/categories_100k.jsonl", lines=True)


# EDA 

# type(jsonObj)
jsonObj.shape
jsonObj.head(2)
len(jsonObj.title.unique())
type(jsonObj.title.duplicated())
bool_series = jsonObj["title"].duplicated()
bool_series
jsonObj[bool_series][10:20]

all_cats= [str.split(jsonObj.categories[i]) for i in range(len(jsonObj.categories))]
all_cats=[set(all_cats[i]) for i in range(len(jsonObj.categories))]
[(all_cats, type(all_cats)) for all_cats in all_cats[0:5]]
total_set = set()
total_set.update(*all_cats)

len(total_set)
total_set
total_list=list(total_set)


df_new = jsonObj.set_index('title',drop=True, append=False, inplace=False, verify_integrity=False)
df_new
df_new02 = jsonObj.reset_index(drop=False)
df_new02
df_new['categories'] = df_new['categories'].str.split(" ")
df_new

# convert to list
jsonObj['categories'] = jsonObj['categories'].str.split(" ")

# convert list of pd.Series then stack it
new_df = (jsonObj
 .set_index(["title"])['categories']
 .apply(pd.Series)
 .stack()
 .reset_index()
 .drop('level_1', axis=1)
 .rename(columns={0:'categories'}))
 
 
enc = preprocessing.OneHotEncoder(dtype=np.bool8)
enc = enc.fit(new_df[["categories"]])
#enc.categories_
cats = [name.strip("x0_") for name in enc.get_feature_names()]
enc.get_feature_names()
binary_cats = enc.transform(new_df[["categories"]])
new_df[["categories"]]
type(binary_cats)
# We can convert this to a normal numpy array
# type(binary_cats.toarray())
binary_cats.toarray()
jsonObj_bin = pd.concat([new_df.drop(["categories"],axis=1),
pd.DataFrame(binary_cats.toarray())],axis=1) # [column for column in jsonObj_bin]
jsonObj_bin.shape
jsonObj_bin[jsonObj_bin.title=="Masers and star formation"].agg(np.any) [1:-1].sum()
tran_df = jsonObj_bin.groupby("title").agg(np.any) jsonObj_bin.shape,len(jsonObj_bin.title.unique()), tran_df.shape tran_df.columns=cats
tran_df.head()
tran_df.sum(axis=1).hist()
tran_df.sum(axis=1).describe()
tran_df.astype(np.int16).describe()

# association rules analysis
from mlxtend.frequent_patterns import apriori,association_rules,fpgrowth
apriori(tran_df,min_support=0.02,use_colnames=True).unstack().unstack()
apriori(tran_df,min_support=0.02,use_colnames=True)
with pd.option_context('display.max_rows', None, 
                       'display.max_columns', None):  
    print(apriori(tran_df,min_support=0.02,use_colnames=True))
    
freq_cats = apriori(tran_df,min_support=0.02,use_colnames=True)
freq_cats.sort_values("support",ascending=False)
freq_cats = apriori(tran_df,min_support=0.02,use_colnames=True)
association_rules(freq_cats, metric="confidence", min_threshold=0.2)
association_rules(freq_cats, metric="lift", min_threshold=0.2)

with pd.option_context('display.max_rows', None, 
                       'display.max_columns', None):  
    print(association_rules(freq_cats, metric="confidence", min_threshold=0.2))
    
%%timeit
apriori(tran_df,min_support=0.02,use_colnames=True)   
%%timeit
fpgrowth(tran_df,min_support=0.02,use_colnames=True)    
    

# Natural language processing analysis

import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import heapq

jsonObj_abstract = pd.read_json(path_or_buf="/Users/yimin/project/dataMining/abstracts.jsonl", lines=True)
# type(jsonObj)
jsonObj_abstract.shape
[column for column in jsonObj_abstract]
jsonObj_abstract.categories.unique()

pars = [Obstract[i].split("\n") for i in range(len(Obstract))]
total_pars = len(pars)

sent_tokens = []
for i in range(total_pars):
    sent_tokens_be= [sent_tokenize(par) for par in pars[i]]
    sent_tokens.append(sent_tokens_be)
    
def flatten_list(nested_list):
    return [inner_list for sublist in nested_list 
                       for inner_list in sublist]

flat_sents =[flatten_list(sent_tokens_eachDocument) for sent_tokens_eachDocument in sent_tokens]
n_flat_sents = len(flat_sents)

len(flat_sents[0])
n_each_pars = [len(x) for x in flat_sents]
plt.hist(n_each_pars)

sent_words = []
for i in range(len(flat_sents)):
    sent_words_be = [word_tokenize(sent) for sent in flat_sents[i]]
    sent_words.append(sent_words_be)
    
from nltk import pos_tag_sents, pos_tag
tagged_sents = []
for i in range(len(sent_words)):
    tagged_sents_be = pos_tag_sents(sent_words[i])
    tagged_sents.append(tagged_sents_be)

import re
re_tokens = []
for i in range(n_flat_sents):
    re_tokens_be = [re.findall("[^\d\W]+", sent) for sent in flat_sents[i]]
    re_tokens.append(re_tokens_be)
    
flat_words = []
for i in range(len(re_tokens)):
    flat_words_be = flatten_list(re_tokens[i])
    flat_words.append(flat_words_be) 
    
lower_words = []
for i in range(len(flat_words)):
    lower_words_be = [word.lower() for word in flat_words[i]]
    lower_words.append(lower_words_be)

from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
filtered_words = []
for i in range(len(lower_words)):
    filtered_words_be = [word for word in lower_words[i] if word not in stops]
    filtered_words.append(filtered_words_be)
    
stemmer = SnowballStemmer(language='english')
count_stemmed = []
for i in range(len(filtered_words)):
    stemmed_words = [stemmer.stem(word) for word in filtered_words[i]]
    count_stemmed_be = FreqDist(stemmed_words)
    count_stemmed.append(count_stemmed_be)

count_list = [Counter(count_stemmed[i]) for i in range(len(count_stemmed))]
doc_mat = pd.DataFrame.from_records(count_list)
doc_mat=doc_mat.fillna(0).astype(int)
rowsums = doc_mat.sum(axis=1)
rowsums.hist()

tf_df = doc_mat.div(rowsums, axis=0)
doc_freq = doc_mat.aggregate(lambda x: sum(x>0),axis=0) # apply to columns
n_docs = doc_mat.shape[0]

idf = np.log(n_docs/doc_freq)
idf.min(),idf.max()
tf_idf = tf_df.multiply(idf, axis=1)

# Lets see if we can find a document in our corpus which is similar in some sense to the the first paper
with open("/Users/yimin/project/dataMining/Q1_abstract.txt", "r") as f:
    Q1_data = f.read().split(".")

    print(Q1_data)
    
words = [word.lower() for word in re.findall("[^\d\W]+", Q1_data)]
lower_words = [word.lower() for word in words]
Q1_filtered_words = [word for word in lower_words if word not in stops]
Q1_stemmed_words = [stemmer.stem(word) for word in Q1_filtered_words]
Q1_count_stemmed = FreqDist(Q1_stemmed_words)

Q1_df = pd.DataFrame.from_records([Counter(Q1_count_stemmed)])
intersec_words = tf_idf.columns.intersection(Q1_df.columns)
row_count = Q1_df[intersec_words].sum(axis=1)[0]
tf_Q1_df  = Q1_df[intersec_words].div(row_count)

idf_Q1_df = idf[intersec_words]
tf_idf_Q1_df = tf_Q1_df.multiply(idf_Q1_df, axis=1)
doc_intersect = tf_idf[intersec_words]

cos_sim = cosine_similarity(tf_idf_Q1_df, doc_intersect)

cos_sim.max()
cos_sim.min()

np.argsort(-cos_sim)
np.argsort(-cos_sim)[0][0:5]
print(heapq.nlargest(5, list_num))


# Topic modelling


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from plotnine import *
import matplotlib.pyplot as plt

def extract_words(text):
    words = [word.lower()
             for word in re.findall("[^\d\W]+", text)]
    return words

def filter_words(words, stops, stemmer):
    filtered_words = [word for word in words 
                      if word not in stops]
    stemmed_words = [stemmer.stem(word) 
                     for word in filtered_words]
    return stemmed_words


def tokenise(text, stops, stemmer):
    return filter_words(extract_words(text), stops, stemmer)
    
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer(language='english')

from nltk.corpus import stopwords
stops = stopwords.words('english')

from functools import partial
tokeniser_partial = partial(tokenise, stemmer=stemmer, stops=stops)

vectorizer = CountVectorizer(analyzer=tokeniser_partial)
DD = vectorizer.fit_transform(Obstract)
# this process needs time to run

tfidf_array = TfidfTransformer().fit_transform(DD)
DD = tfidf_array

from sklearn.decomposition import LatentDirichletAllocation as LDA

lda = LDA(n_components = 5, max_iter=50,
          topic_word_prior=0.1,
          doc_topic_prior=0.1,
          random_state=10)
          
lda.fit(DD)
lda.components_.shape
# take more time

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

terms = vectorizer.get_feature_names()
plot_top_words(lda, terms, 10, 'Topics in LDA model')

DD_dash = lda.transform(DD)
DD_dash[0,:]


from sklearn.decomposition import TruncatedSVD
lsa_obj = TruncatedSVD(n_components=5).fit(DD)
from sklearn.decomposition import NMF
nmf = NMF(n_components = 5,random_state=0)
nmf.fit(DD)
nmf_term_df = pd.DataFrame(nmf.components_.T)
nmf_term_df["terms"] = terms
"   ".join(nmf_term_df.sort_values(0, ascending=False).terms.head(10))
"   ".join(nmf_term_df.sort_values(1, ascending=False).terms.head(10))
"   ".join(nmf_term_df.sort_values(2, ascending=False).terms.head(10))
"   ".join(nmf_term_df.sort_values(3, ascending=False).terms.head(10))
"   ".join(nmf_term_df.sort_values(4, ascending=False).terms.head(10))

x=np.argmax(DD_dash,axis=1).tolist()
x=[int(x[i])+1 for i in range(len(x))]

Topic_df=pd.DataFrame(DD_dash)
Topic_df["Topic_label"]=pd.DataFrame(x)
Topic_df.columns =  ["topic" + str(x) for x in  range(1,6)]+["Topic_label"]
Topic_df["Cats"] = jsonObj_abstract.categories
Topic_df[Topic_df.Topic_label==1]
Topic_df[Topic_df.Topic_label==4].groupby("Cats").count()


fig= sns.countplot(x="Topic_label",hue="Cats",data=Topic_df,palette="magma")
fig = fig.get_figure()
fig.savefig('hist.png')


Topic_df.Cats.unique()
map_dict = {'hep-ph':3, 'cond-mat.mtrl-sci':4, 'astro-ph':2, 'math.AG':1, 'quant-ph':5}
Topic_df["ref_lab"] = [map_dict[x] for x in Topic_df["Cats"] ]

from sklearn.metrics import accuracy_score
y_pred = Topic_df["Topic_label"]
y_true = Topic_df["ref_lab"]
accuracy_score(y_true, y_pred)

# 0.8521811325976407




