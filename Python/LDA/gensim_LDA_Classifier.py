#
#GENSIM Classifier

#MODEL='M_201028_FU_23'
MODEL='M_201028_FU_M23_T22'
PATH='./Results/'

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
#Full DB
#query = "SELECT Id AS Id, Text_Lemmatized AS Text_Lemmatized FROM jenny.articles WHERE DOI != ''  AND `Type` = 'journal-article' LIMIT " + str(LIMIT) + ";"
#Subtopic
query = "SELECT Id AS Id, Text_Lemmatized AS Text_Lemmatized FROM jenny.articles WHERE  M_201028_FU_23_T = 22 AND DOI != ''  AND `Type` = 'journal-article' LIMIT " + str(LIMIT) + ";"

df = pd.read_sql(query, connection)


#load LDA model
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
lda_model = gensim.models.LdaMulticore.load(PATH+MODEL+'_lda.model')
id2word = Dictionary.load(PATH+MODEL+'_lda.id2word')
#View the topics in the model
from pprint import pprint
pprint(lda_model.print_topics())
print(" ---")
#pprint(lda_model.print_topic(0))

#Classifier
#https://stackoverflow.com/questions/61198009/classify-text-with-gensim-lda-model

counter = 0
for index, row in df.iterrows():
    #print(str(row['Id']) + ' ' + str(row['Text_Lemmatized']))

    record_id = row['Id']
    text_string = row['Text_Lemmatized']

    wordlist = list(text_string.split(" "))
    line_bow = id2word.doc2bow(wordlist)
    doc_lda = lda_model[line_bow]
    print("ID " +str(record_id)+ " Classified to:")
    #print(doc_lda[0])
    match=max(doc_lda[0],key=lambda item:item[1])
    topic=match[0] + 1 #Correction for LDAvis
    probability=match[1]
    print("Assigned topic " + str( match ) )

    #Update record in DB
    query = ("UPDATE articles SET " +MODEL+ "_T = %s, " +MODEL+ "_P = %s WHERE  Id = %s")
    cursor.execute(query, (str(topic), str(probability), str(record_id)))

    connection.commit()
    print(counter)
    counter += 1
    print("----")


if (connection.is_connected()):
    connection.close()
    cursor.close()
    print("MySQL connection is closed")
