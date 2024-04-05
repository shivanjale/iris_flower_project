import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_iris

df_iris=load_iris()
x,y=load_iris(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)

model=LogisticRegression()
model.fit(x_train,y_train)

training_accuracy=model.score(x_train,y_train)
testing_accuracy=model.score(x_test,y_test)

print("Model Training Accuracy: ",training_accuracy)
print("Model Testing Accuracy: ",testing_accuracy)

pickle.dump(model,open("logistic_model.pkl","wb"))



