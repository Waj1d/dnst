import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix , plot_confusion_matrix, precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
data = pd.read_csv("CipmaDNSDataset2022.csv", header = None)
X = data.drop(data.columns[[3, 8]], axis=1)
Y = data[8]

##########                        SMOTE                   #################
oversample = SMOTE()
X, Y = oversample.fit_resample(X,   Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.5)

##########                        Training               #################

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

##################                Evaluation              ##############
accuracy = accuracy_score(y_test,y_pred)*100
print(accuracy)
print(confusion_matrix(y_test,y_pred))
print('Precision = ', precision_score(y_test, y_pred, average='binary')*100)
print('Recall = ', recall_score(y_test, y_pred, average='binary')*100)
print('F1 Score = ', f1_score(y_test, y_pred, average='binary')*100)