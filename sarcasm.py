#dataset from https://www.kaggle.com/datasets/danofer/sarcasm
#followed programming/ML tutorial from https://thecleverprogrammer.com/2021/08/24/sarcasm-detection-with-machine-learning/



import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

filename = 'sarcasm6.csv' #put the path to the .csv with the data here
train_df = pd.read_csv(filename)

train_df.dropna(subset=['comment'], inplace=True)

train_df["label"] = train_df["label"].map({0: "No Sarcasm Detected", 1: "Sarcasm Detected!"})


cv = CountVectorizer()
X = cv.fit_transform(train_df['comment'])

train_texts, valid_texts, y_train, y_valid = train_test_split(X, train_df['label'], random_state=17)

model = BernoulliNB() #Uses the Bernoulli Naive-Bayes algorithm
model.fit(train_texts, y_train)
#print(model.score(valid_texts, y_valid))

user = input("Enter a text: ")
data = cv.transform([user]).toarray()
listoutput = model.predict(data)
output = ''.join(listoutput)
print(output)
