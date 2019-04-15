

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#preprocessinhg the data
dataset=pd.read_csv('transfusion.csv') #loading the dataset
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

dataset.isna().sum() #no missing data


#columns
dataset.columns
#seeing the heat map
sns.heatmap(dataset.corr())
plt.show()
#changing columns names
dataset.columns=['Recency', 'Frequency', 'Monetaryblood', 'Times', 'Donateddata']

#checking the donated data variable  0's and 1's
dataset.Donateddata.value_counts()

#filter out the applicants that donated
donated = dataset.loc[y == 1]
#filter out the applicants that not donated
not_donated = dataset.loc[y == 0]

#plots for donated and not donated
plt.scatter(donated.iloc[:, 0], donated.iloc[:, 1], s=10, label='donated')
plt.scatter(not_donated.iloc[:, 0], not_donated.iloc[:, 1], s=10, label='Not donated')
plt.legend()
plt.show()

#splittingataset and testing dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=90)

#fitting to the logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

classifier = LogisticRegression(solver='lbfgs')
data = DecisionTreeClassifier(criterion='entropy', random_state=85)

classifier.fit(x_train, y_train)
data.fit(x_train, y_train)

#predicting the results
y_predict = classifier.predict(x_test)
y_pred=data.predict(x_test)



#Makig the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predict)
cm1=confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)   #logistic regresion
accuracy_score(y_test, y_pred)    #decision tree


#classification report of the datset
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
print(classification_report(y_test, y_pred))

