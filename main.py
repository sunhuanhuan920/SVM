import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# read csv file
df = pd.read_csv("./Data/emails.csv")

# separate X and y from data frame
X = np.array(df.iloc[:, 1:3001])
y = df.iloc[:, 3001].values

# split data set to training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# scikit-learn SVM
# clf = SVC(kernel='poly')
# clf.fit(X_train, y_train)
# y_predict = clf.predict(X_test)
# print("Accuracy Score for SVM (scikit-learn implementation): ", accuracy_score(y_predict, y_test))
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print("Accuracy Score for SVM (scikit-learn implementation): ", accuracy_score(y_predict, y_test))