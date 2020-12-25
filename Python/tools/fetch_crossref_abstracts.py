from crossref.restful import Works
works = Works()

from bs4 import BeautifulSoup

import time

import mysql.connector
from mysql.connector import Error

try:
    connection = mysql.connector.connect(host='localhost',
                                         database='jenny',
                                         user='myusername',
                                         password='mypassword')
    cursor = connection.cursor()

except Error as e:
    print("Error reading data from MySQL table", e)

#Fetch the rows from the DB
number_of_rows = cursor.execute("select Id, DOI from articles WHERE DOI != '' AND Id > 23440");
result = cursor.fetchall()

for row in result:
    id = row[0]
    doi = row[1]

    raw = works.doi(doi)
    title = raw['title'][0]
    type = raw['type']
    #check if key abstract exists
    if "abstract" in raw:
        cleanabstract = BeautifulSoup(raw['abstract'], "lxml").text
    else:
        cleanabstract = ""

    print("----------------------------------------------------", id)
    print (title)
    print (type)
    print (cleanabstract)

    query = ("UPDATE articles SET Title = %s , Type = %s , CR_Abstract = %s"
             "WHERE  Id = %s")
    cursor.execute(query, (title, type, cleanabstract, id))
    connection.commit()

    time.sleep(1)

if (connection.is_connected()):
    connection.commit()
    connection.close()
    cursor.close()
    print("MySQL connection is closed")
