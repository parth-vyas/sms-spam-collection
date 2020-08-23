# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:14:51 2020

@author: PARTH
"""

import pandas as pd

message=pd.read_csv('G:\DataScienceProject\smsspamcollection\SMSSpamCollection', sep='\t' , names=["label","text"])

#data cleaning and preprocesssing

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
import re
ps = PorterStemmer()
corpus=[]

for i in range(0,len(message)):  
    data = re.sub('[^A-Za-z]',' ',message["text"][i])
    data = data.lower()
    data = data.split()
    data = [ps.stem(word) for word in data if not word in stopwords.words('English')]  
    data = ' '.join(data)
    corpus.append(data)
    
corpus    

#bag of words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = pd.get_dummies(message["label"])
y = y.loc[:,["ham"]]

#train model and predict

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
model = GaussianNB()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)


