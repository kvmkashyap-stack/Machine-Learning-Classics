import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
  
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    
    df = pd.DataFrame(X, columns=cancer.feature_names)
    df["target"] = y

    sns.pairplot(df, hue="target")
    plt.show()
  

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    
    y_pred = knn.predict(X_test)

   
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=cancer.target_names,
        yticklabels=cancer.target_names
    )
    plt.show()


if __name__ == "__main__":
    main()
