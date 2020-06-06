# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

# Importing the dataset
dataset_test = pd.read_csv('datasets_test.csv')
dataset_train = pd.read_csv('datasets_train.csv')

# Droping Unnessecary column
dataset_train.drop(['ID'], axis=1, inplace=True)
dataset_test.drop(['ID'], axis=1, inplace=True)

# Cleaning the texts
corpus_train = []
for i in range(0, 5287):
    tweet = re.sub('[^a-zA-Z]', ' ', dataset_train['text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus_train.append(tweet)
    
corpus_test = []
for i in range(0, 1962):
    tweet = re.sub('[^a-zA-Z]', ' ', dataset_test['text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus_test.append(tweet)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X_train = cv.fit_transform(corpus_train).toarray()
X_test = cv.fit_transform(corpus_test).toarray()
y_train = dataset_train.iloc[:, 1].values

pickle.dump(cv, open('cv.pkl', 'wb'))

# Training the model on random forest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy')
classifier.fit(X_train, y_train)

pickle.dump(classifier, open('classifier.pkl', 'wb'))

# Predicting the Train set results
y_pred_train = classifier.predict(X_train)

# Making the Confusion Matrix
print('Confusion Matrix :')
print(confusion_matrix(y_train, y_pred_train)) 
print('Accuracy Score :',accuracy_score(y_train, y_pred_train))
print('Report : ')
print(classification_report(y_train, y_pred_train))

y_pred_test = classifier.predict(X_test)