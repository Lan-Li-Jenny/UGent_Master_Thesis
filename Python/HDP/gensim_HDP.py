#
#GENSIM HDP model


#Fix ssl errors in wordnet downloads
import ssl
from pprint import pprint

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import pandas as pd
import mysql.connector
from mysql.connector import Error

#
# Fetch data from SQL
try:
    connection = mysql.connector.connect(host='192.168.xxx.xxx',
                                         database='jenny',
                                         user='myusername',
                                         password='mypassword')
    cursor = connection.cursor()

except Error as e:
    print("Error reading data from MySQL table", e)

LIMIT = 100000

#Fetch the rows from the DB
query = "SELECT Title as Title, Abstract AS Abstract, CR_Abstract as CR_Abstract FROM jenny.articles WHERE DOI != ''  AND `Type` = 'journal-article' LIMIT " + str(LIMIT) + ";"
df = pd.read_sql(query, connection)

if (connection.is_connected()):
    connection.close()
    cursor.close()
    print("MySQL connection is closed")

#Merge the google abstracts into the crossref abstracts if no abstract available
df.loc[df['CR_Abstract'] == '','CR_Abstract'] = df['Abstract']

#Combine title and abstract
df['Text'] = df['Title'] + " " + df['CR_Abstract']
#df['Text'] = df['Title']

#Filter everything to marketing and entrepreneurial
df = df[df['Text'].str.contains("marketing") & df['Text'].str.contains("entrepreneurial")]

# Remove words between brackets
#df['Text'] = df['Text'].str.replace(r"(\s*\(.*?\)\s*)", " ").str.strip()
df['Text'] = df['Text'].str.replace("\\xa0...", "").str.strip()

#
# Remove punctuation/lower casing
import re
# Remove punctuation
df['Text_processed'] = df['Text'].map(lambda x: re.sub('[,\.!?:]', '', x))
# Convert the titles to lowercase
df['Text_processed'] = df['Text_processed'].map(lambda x: x.lower())
# Print out the first rows of papers
df['Text_processed'].head()

#
#Tokenize words and further clean-up text
import gensim
from gensim.utils import simple_preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data = df.Text_processed.values.tolist()
data_words = list(sent_to_words(data))

#
#Phrase Modeling: Bi-grams and Tri-grams
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

#
#Remove Stopwords and Lemmatize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram[bigram[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#Call the functions
import spacy
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])  #Unigrams
#data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']) #Bigrams
print(data_lemmatized[:1])

#
#Data Transformation: Corpus and Dictionary
import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1])

#
#Base HDP Model
print('Running hdp model...')
from gensim.models import HdpModel
hdpmod = HdpModel(corpus=corpus,
                  id2word=id2word,
                  random_state=100,
                  #chunksize=100,
                  K=15, #15
                  T=150 #150
                  )

from pprint import pprint
pprint(hdpmod.print_topics(num_topics=1000, num_words=10))

from gensim.models import CoherenceModel
coherence_model_hdp = CoherenceModel(model=hdpmod, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_hdp = coherence_model_hdp.get_coherence()
print('\nCoherence Score: ', coherence_hdp)



#Convert to an LDA model
hdp_lda = hdpmod.suggested_lda_model()

import pyLDAvis.gensim
import pickle
import pyLDAvis
LDAvis_prepared = pyLDAvis.gensim.prepare(hdpmod,  corpus, id2word, sort_topics=False)
pyLDAvis.save_html(LDAvis_prepared, 'hdp_unigrams_title_abstract.html')
