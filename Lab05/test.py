import pandas as pd
import numpy as np
from scipy.spatial.distance import minkowski
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign")

features = [
    "Year_Birth",
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "Response"
]

data = data[features]

data = data.dropna(subset=["Response"])

for col in data.columns:
    data[col] = data[col].fillna(data[col].median())

X = data.drop("Response",axis=1).values
Y = data["Response"].values

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.3,random_state=42)

def predict(Xtrain,Ytrain,Xtest,k):
    predictions=[]

    for testpoint in Xtest:
        distances=[]

        for i in range(len(Xtrain)):
            d = minkowski(Xtrain[i],testpoint,p=2)
            distances.append((d,Ytrain[i]))

        distances.sort()

        neighbors = distances[:k]

        votes={}

        for d, label in neighbors:

            if label not in votes:
                votes[label] = 0

            votes[label] +=1 #votes[label] += 1 / (d + 1e-10)

        prediction = max(votes,key=votes.get)
        predictions.append(prediction)

    return np.array(predictions)

def accuracy(y_true, y_pred):

    return np.mean(y_true == y_pred)

k = 3

predictions = predict(
    Xtrain,
    Ytrain,
    Xtest,
    k
)

acc = accuracy(Ytest, predictions)

print("Our kNN Accuracy:", acc)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(Xtrain,Ytrain)
sklearnpred = model.predict(Xtest)
sklearnacc = model.score(Xtest,Ytest)

print("Sklearn kNN Accuracy:", sklearnacc)
