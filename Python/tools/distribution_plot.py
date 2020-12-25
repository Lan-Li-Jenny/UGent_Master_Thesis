#
#Distribution chart plot

#MODEL='M_201028_FU_23'
MODEL='M_201028_FU_M10'
TOPIC='1'

import pandas as pd
import numpy as np
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
#query = "SELECT Id AS Id, M_201028_FU_10_P AS P FROM jenny.articles WHERE DOI != ''  AND `Type` = 'journal-article' LIMIT " + str(LIMIT) + ";"
#Subtopic
query = "SELECT Id AS Id, M_201028_FU_10_P AS P FROM jenny.articles WHERE  M_201028_FU_10_T = 1 AND DOI != ''  AND `Type` = 'journal-article' LIMIT " + str(LIMIT) + ";"

df = pd.read_sql(query, connection)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sns.set_style("darkgrid")
# seaborn histogram
sns.distplot(df['P'], hist=True, kde=False,
             bins=100, color = 'blue',
             hist_kws={'edgecolor':'black'})

# Add labels
plt.title('Match probabilities, model= ' + MODEL + ' , T= ' + TOPIC)
plt.xlabel('Probability')
plt.ylabel('Articles')

# Add total, Mean, Std
total = len(df)
mean = round(df['P'].mean(),4)
stdev = round(df['P'].std(),4)
plt.text(0.3, 20, ' Total #: \n P Mean: \n P Std: ',
         horizontalalignment='left', alpha=0.7,  fontsize=9)
plt.text(0.4, 20, '' + str(total) + '\n' + str(mean) + '\n' + str(stdev),
         horizontalalignment='right', alpha=0.7,  fontsize=9)



plt.tight_layout()
plt.show()

# Density Plot and Histogram of all arrival delays
#sns.distplot(df['P'], hist=True, kde=True,
#             bins=int(180/5), color = 'darkblue',
#             hist_kws={'edgecolor':'black'},
#             kde_kws={'linewidth': 4})

# Add labels
#plt.title('Match probabilities model: ' + MODEL + ' , T= ' + TOPIC)
#plt.xlabel('Probability')
#plt.ylabel('Articles')

#plt.tight_layout()
#plt.show()

#http://www.futurile.net/2016/03/01/text-handling-in-matplotlib/
