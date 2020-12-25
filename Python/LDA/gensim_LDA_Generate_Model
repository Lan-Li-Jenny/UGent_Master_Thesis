#
#GENSIM OPTIMIZED RUN
#Use the parqmeters generqted by gensim1

PATH='./Results/'
#MODEL='M_201028_FU_50'
#MODEL='M_201028_FU_M23_T22'
MODEL='test'
FILE=PATH + MODEL + "_"

# Set parameters.
num_topics = 200
chunksize = 1000    #Number of documents to be used in each training chunk. 500
passes = 80         #Number of passes through the corpus during training. 20
iterations = 800    #Maximum number of iterations through the corpus 400
eval_every = 1      #Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
decay=0.5           #What percentage of the previous lambda value is forgotten when each new document is examined
alpha=50/num_topics             #def 1.0 Typ 50/K #A higher alpha gives a more dense distribution whereas a lower alpha gives a more sparse distribution. The catch of using a low alpha? That really depends, but the most immediate consequence is that you force documents to contain only 1 or a few topics, while there might in fact be more of a mixture of topics. Being able to identify multiple topics within a document is also one the strengst of LDA
beta=0.1           #def 1.0 typ 0.1 or 200/W #A lower beta means that each topic has fewer words that are strong indicators of the topic, whereas a higher beta means that the PMF of each topic is more spread out over multiple words. Theoretically, the lowest beta would mean that each topic is only related to one word, whereas the highest beta means that the PMF is uniform (no word more likely than another).
minimum_probability= 0.01 #minimum_probability (float, optional) – Topics with a probability lower than this threshold will be filtered out. 0.01
#alpha, beta: Griffiths TL, Steyvers M (2004). “Finding Scientific Topics.” Proceedings of the National Academy of Sciences of the United States of America, 101, 5228–5235.

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

print("Fetching data...")
#
# Fetch data from SQL
try:
    connection = mysql.connector.connect(host='192.168.xxx.xxx',
                                         database='jenny',
                                         user='myusername',
                                         password='mypqssword')
    cursor = connection.cursor()

except Error as e:
    print("Error reading data from MySQL table", e)

LIMIT = 100000

#Fetch the rows from the DB
#Full topics
query = "SELECT Title as Title, Abstract AS Abstract, CR_Abstract as CR_Abstract FROM jenny.articles WHERE DOI != ''  AND `Type` = 'journal-article' LIMIT " + str(LIMIT) + ";"
#Subtopics
#query = "SELECT Title as Title, Abstract AS Abstract, CR_Abstract as CR_Abstract FROM jenny.articles WHERE M_201028_FU_23_T = 22  AND `Type` = 'journal-article' LIMIT " + str(LIMIT) + ";"

df = pd.read_sql(query, connection)

if (connection.is_connected()):
    connection.close()
    cursor.close()
    print("MySQL connection is closed")

print("Lemmatizing...")

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

#stop_words.extend(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m','n','o','p','q','r','s','t', 'u', 'v', 'w', 'x', 'y', 'z',
#                   "about", "across", "after", "all", "also", "an", "and", "another", "added",
#                   "any", "are", "as", "at", "basically", "be", "because", 'become', "been", "before", "being",
#                   "between","both", "but", "by","came","can","come","could","did","do","does","each","else",
#                   "every","either","especially", "for","from","get","given","gets", 'give','gives',"got","goes","had",
#                   "has","have","he","her","here","him","himself","his","how","if","in","into","is","it","its","just",
#                   "lands","like","make","making", "made", "many","may","me","might","more","most","much","must","my",
#                   "never","provide", "provides", "perhaps","no","now","of","on","only","or","other", "our","out","over",
#                   "re","said","same","see","should","since","so","some","still","such","seeing", "see", "take","than",
#                   "that","the","their","them","then","there", "these","they","this","those","through","to","too",
#                   "under","up","use","using","used", "underway", "very","want","was","way","we","well","were","what",
#                   "when","where","which","while","whilst","who","will","with","would","you","your", 'etc', 'via', 'eg', 'edu'])

gist_file = open("./STOPWORDS/gist_stopwords.txt", "r")
content = gist_file.read()
stop_words_imp = content.split(",")
gist_file.close()
stop_words.extend(stop_words_imp)

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram[bigram[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): #https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/#6lemmatization
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
#nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])  #Unigrams
#data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']) #Bigrams

#print(data_lemmatized[:1])

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
#print(corpus[:1])

print("Running LDA...")
#Best training values
#https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/

#
#Base LDA Model #LdaMulticore #LdaModel
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics,
                                       decay=decay,
                                       alpha=alpha,
                                       eta=beta,
                                       minimum_probability=minimum_probability,
                                       iterations=iterations,
                                       random_state=100,
                                       chunksize=chunksize,
                                       #eval_every=eval_every,
                                       passes=passes,
                                       per_word_topics=True)
print("Post processing...")
#View the topics in the model
from pprint import pprint
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())

#Save them model to disk
lda_model.save(FILE+'lda.model')
id2word.save(FILE+'lda.id2word')

#Save model topics sorted
top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.extend([(t,) + x for x in lda_model.show_topic(t, topn=30)])
top_words = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P'])
#top_words['Topic'] + 1 #correct to match LDAvis
top_words.to_csv(FILE+'lda_top_words.csv')

#Compute Model Perplexity and Coherence Score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_c_v = coherence_model_lda.get_coherence()
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_npmi')
coherence_npmi = coherence_model_lda.get_coherence()
print('\nCoherence Score c_v: ', coherence_c_v)
print('\nCoherence Score npmi: ', coherence_npmi)
perplexity_lda = lda_model.log_perplexity(corpus)
print('\nPerplexity Score: ', perplexity_lda)

#Save Model parameters
model_params = {'num_topics': [],
                'chunksize': [],
                'passes': [],
                'iterations': [],
                'eval_every': [],
                'decay': [],
                'alpha': [],
                'beta': [],
                'minimum_probability': [],
                'Coherence_c_v': [],
                'Coherence_npmi': [],
                'Perplexity': []
                 }

model_params['num_topics'].append(num_topics)
model_params['chunksize'].append(chunksize)
model_params['passes'].append(passes)
model_params['iterations'].append(iterations)
model_params['eval_every'].append(eval_every)
model_params['decay'].append(decay)
model_params['alpha'].append(alpha)
model_params['beta'].append(beta)
model_params['minimum_probability'].append(minimum_probability)
model_params['Coherence_c_v'].append(coherence_c_v)
model_params['Coherence_npmi'].append(coherence_npmi)
model_params['Perplexity'].append(perplexity_lda)
pd.DataFrame(model_params).to_csv(FILE+'params.csv', index=False)

#
#Visualize the topics
import pyLDAvis.gensim
import pickle
import pyLDAvis
# Visualize the topics
#doc_lda = lda_model[corpus]
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False)
pyLDAvis.save_html(LDAvis_prepared, FILE+'lda.html')
