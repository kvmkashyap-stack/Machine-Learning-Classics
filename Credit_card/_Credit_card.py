import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("creditcard.csv")

print(df.shape)
print(df.isnull().sum())

df = df.drop_duplicates()
df = df.dropna(subset=["Class"])

scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

X = df.drop("Class", axis=1)
y = df["Class"]

sample_df = df.sample(5000)
sns.pairplot(sample_df[["V1","V2","V3","V4","Amount","Class"]], hue="Class")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("LOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("RANDOM FOREST RESULTS")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


tsne = TSNE(n_components=3, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X.sample(3000)) 
y_sample = y.loc[X.sample(3000).index]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    X_tsne[:,0],
    X_tsne[:,1],
    X_tsne[:,2],
    c=y_sample,
    cmap='coolwarm',
    alpha=0.7
)

ax.set_xlabel("TSNE-1")
ax.set_ylabel("TSNE-2")
ax.set_zlabel("TSNE-3")
plt.title("3D t-SNE Visualization - Credit Card Fraud")
plt.show()


sample = X_test.iloc[[0]]
pred = rf.predict(sample)
print("Normal test prediction:", "Fraud" if pred[0]==1 else "Normal")

fraud_sample = X_test[y_test==1].iloc[[0]]
pred_fraud = rf.predict(fraud_sample)
print("Fraud test prediction:", "Fraud" if pred_fraud[0]==1 else "Normal")
