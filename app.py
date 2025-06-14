from random import random
from xml.parsers.expat import model
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import numpy as np
import nltk
from sklearn.datasets import load_files
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer
import sqlite3
from googletrans import Translator
import warnings

app = Flask(__name__)

translator = Translator()

#Reading data

df= pd.read_csv("data.csv")

Tweet = []
Labels = []

for row in df["Tweets"]:
    #tokenize words
    words = word_tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = set(stopwords.words('english'))
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    Tweet.append(lemma_list)



df['message']=df['Tweets']

#df = df[0:2000]
X = df['message']
y = df['Feeling']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

import joblib
model = joblib.load("model_performance.sav")



@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        
        message = request.form['message']
        translations = translator.translate(message, dest='en')
        message =  translations.text
        data = [message]
        #cv = CountVectorizer()
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)
        
        print(my_prediction[0])
        return render_template('result.html',prediction = my_prediction[0],message=message)

@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
	return render_template("index.html")



if __name__ == '__main__':
	app.run(debug=False)
