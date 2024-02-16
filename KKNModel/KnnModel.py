#importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#load the dataset
df = pd.read_csv('fraud.csv')

#spliting the data into X and y
X = df.drop('fraud', axis=1)
y = df['fraud']

#spliting the data into training and tests sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#training the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#Predictions
y_pred = knn.predict(X_test)

#finding the accuracy
acc = accuracy_score(y_test, y_pred)
print("Acc", acc*100)

#confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix for fraud data', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()








