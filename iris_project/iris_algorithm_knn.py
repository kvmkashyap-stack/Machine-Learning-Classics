import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    df = pd.DataFrame(X, columns=iris.feature_names)
    df["target"] = y

    sns.pairplot(df, hue="target")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )
    plt.show()

    pca=PCA(n_components=3)
    X_train_pca=pca.fit_transform(X_train)

    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")

    scatter=ax.scatter(
        X_train_pca[:,0],
        X_train_pca[:,1],
        X_train_pca[:,2],
        c=y_train,
        cmap="coolwarm"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.title("PCA 3D Analysis")
    plt.show()



if __name__ == "__main__":
    main()
