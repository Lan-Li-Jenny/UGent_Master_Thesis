#
#GENSIM OPTIMIZER
#Saves a CSV file with the parameters
#
#https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0

MODEL='M_201028_FU_M23_T22'

#Fix ssl errors in wordnet downloads
import ssl

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
#Full topics
#query = "SELECT Title as Title, Abstract AS Abstract, CR_Abstract as CR_Abstract FROM jenny.articles WHERE DOI != ''  AND `Type` = 'journal-article' LIMIT " + str(LIMIT) + ";"
#Subtopics
query = "SELECT Title as Title, Abstract AS Abstract, CR_Abstract as CR_Abstract FROM jenny.articles WHERE M_201028_FU_23_T = 22  AND `Type` = 'journal-article' LIMIT " + str(LIMIT) + ";"

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
#df = df[df['Text'].str.contains("marketing") & df['Text'].str.contains("entrepreneurial")]

# Remove words between brackets
df['Text'] = df['Text'].str.replace(r"(\s*\(.*?\)\s*)", " ").str.strip()
#df['Text'] = df['Text'].str.replace("\\xa0...", "").str.strip()

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

stop_words.extend(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m','n','o','p','q','r','s','t', 'u', 'v', 'w', 'x', 'y', 'z',
                   "about", "across", "after", "all", "also", "an", "and", "another", "added",
                   "any", "are", "as", "at", "basically", "be", "because", 'become', "been", "before", "being",
                   "between","both", "but", "by","came","can","come","could","did","do","does","each","else",
                   "every","either","especially", "for","from","get","given","gets", 'give','gives',"got","goes","had",
                   "has","have","he","her","here","him","himself","his","how","if","in","into","is","it","its","just",
                   "lands","like","make","making", "made", "many","may","me","might","more","most","much","must","my",
                   "never","provide", "provides", "perhaps","no","now","of","on","only","or","other", "our","out","over",
                   "re","said","same","see","should","since","so","some","still","such","seeing", "see", "take","than",
                   "that","the","their","them","then","there", "these","they","this","those","through","to","too",
                   "under","up","use","using","used", "underway", "very","want","was","way","we","well","were","what",
                   "when","where","which","while","whilst","who","will","with","would","you","your", 'etc', 'via', 'eg', 'edu'])

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
#Base LDA Model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10,
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)
#View the topics in the model
from pprint import pprint
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

#Compute Model Perplexity and Coherence Score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_npmi')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# get the perplexity score
perplexity_lda = lda_model.log_perplexity(corpus)
print('\nPerplexity Score: ', perplexity_lda)


#
#Hyperparameter Tuning

# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=1000,
                                           passes=80,
                                           iterations=800,
                                           decay=1,
                                           alpha=a,
                                           eta=b,
                                           minimum_probability = 0.01,
                                           per_word_topics=True)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_npmi')
    perplexity_model_lda = lda_model.log_perplexity(corpus)

    return coherence_model_lda.get_coherence(), perplexity_model_lda

#Iterate over model parameters
import numpy as np
import tqdm


grid = {}
grid['Validation_Set'] = {}
# Topics range
min_topics = 1 #4
max_topics = 20 #40
step_size = 1 #2
topics_range = range(min_topics, max_topics+step_size, step_size)
# Alpha parameter
#alpha = list(np.arange(1, 2, 1))
alpha = [5] #1
#alpha.append('symmetric')
#alpha.append('asymmetric')
# Beta parameter
#beta = list(np.arange(0.01, 0.11, 0.02))
beta = [0.01] #0.01
#beta.append('symmetric')
# Validation sets
num_of_docs = len(corpus)

corpus_sets = [  # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
    # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
    #gensim.utils.ClippedCorpus(corpus, num_of_docs * 0.75),
    corpus]
corpus_title = ['100% Corpus']
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': [],
                 'Perplexity': []
                 }
# Can take a long time to run
if 1 == 1:
    steps=(max_topics-min_topics+1)/min_topics*len(alpha)*len(beta)
    pbar = tqdm.tqdm(total=steps)

    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterate through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv, p = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word,
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    model_results['Perplexity'].append(p)

                    pbar.update(1)
    pd.DataFrame(model_results).to_csv(MODEL + '_tuning.csv', index=False)
    pbar.close()
