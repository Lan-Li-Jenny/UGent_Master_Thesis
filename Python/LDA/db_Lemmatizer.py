#
#LEMMATIZER, preparing DB for classification
#


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
                                         password='mypassword')
    cursor = connection.cursor()

except Error as e:
    print("Error reading data from MySQL table", e)

LIMIT = 100000

#Fetch the rows from the DB
query = "SELECT Id as Id, Title as Title, Abstract AS Abstract, CR_Abstract as CR_Abstract FROM jenny.articles WHERE DOI != ''  AND `Type` = 'journal-article' LIMIT " + str(LIMIT) + ";"
df = pd.read_sql(query, connection)


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
#Remove Stopwords and Lemmatize
import gensim
from gensim.utils import simple_preprocess
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

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

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

print(df['Text_processed'])

for index, row in df.iterrows():
    #print(str(row['Id']) + ' ' + str(row['Text_processed']))

    record_id = str(row['Id'])
    data = row['Text_processed'].split()
    print(data)
    data_words = list(sent_to_words(data))

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    #https: // medium.com / @ tusharsri / remove - add - stop - words - 7e2994c19c67 Spacy vs nltk
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])  # Unigrams
    data_flattened = sum(data_lemmatized, [])
    text_lemmatized= " ".join(map(str, data_flattened))
    print(text_lemmatized)

    #Update record in DB
    query = ("UPDATE articles SET Text_Lemmatized = %s WHERE  Id = %s")
    cursor.execute(query, (text_lemmatized, record_id))
    connection.commit()

    print("---")


if (connection.is_connected()):
    connection.close()
    cursor.close()
    print("MySQL connection is closed")
