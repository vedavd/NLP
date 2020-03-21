# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:51:40 2020

@author: vedav
"""

import pandas as pd
import nltk

#Read the Spam collection Dataset
messages=pd.read_csv('SMSSpamCollection', sep='\t', names=["label","message"])

#Download stopwords
nltk.download('stopwords')


#Importing the required libraries
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#View stopwords in English
stopwords.words('english')

#Create objects of Stemming and Lemmatization
ps=PorterStemmer()
lemma=WordNetLemmatizer()


#Create Corpus
corpus=[]


#View the length of messages i.e. dataset
len(messages)

#Clean the text in the dataset
#including only alphabets,converting all words to lower case,splitting the words
for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

#Create Bag of Words
"""from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()"""
#the above resulted in 6296 unique words. BUt let us consider only top 5000 words


#Create Bag of Words for top 5000 words only
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()

#View the bag of words that is present in X
X

#Check the shape of X
X.shape


#Let us convert the dummy variables present in column 1 containing spam or ham
y=pd.get_dummies(messages['label'])

#Only one column of the dummy variables is necessary due to dummy variable trap
y=y.iloc[:,1].values

#Check what is present in column y
y

#check shape of y
y.shape



#Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


#Training Model using Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train,y_train)

#Predict the results
y_pred=spam_detect_model.predict(X_test)

#Create Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)

#View the confusion Matrix
confusion_m

#Check the accuracy of model
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)


#View the accuracy
accuracy
