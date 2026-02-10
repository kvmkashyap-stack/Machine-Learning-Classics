import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv("churn.csv")

df=df.drop("customerID",axis=1)

df=df.drop_duplicates()
 
le=LabelEncoder()
for col in df.columns:
    if df[col].dtype=="object":
        df[col]=le.fit_transform(df[col])

sns.pairplot(df[['tenure','MonthlyCharges','TotalCharges','Churn']])
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.show()

X=df.drop("Churn",axis=1)
y=df["Churn"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

nb=GaussianNB()
nb.fit(X_train,y_train)
nb_pred=nb.predict(X_test)

print("Naive Bayes Accuracy:",accuracy_score(y_test,nb_pred))
print(confusion_matrix(y_test,nb_pred))
print(classification_report(y_test,nb_pred))

svm=SVC(kernel='rbf')
svm.fit(X_train,y_train)
svm_pred=svm.predict(X_test)

print("SVM Accuracy:",accuracy_score(y_test,svm_pred))
print(confusion_matrix(y_test,svm_pred))
print(classification_report(y_test,svm_pred))
