import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier

# read data
data = pd.read_csv('data_train.csv', index_col="id")
data_test = pd.read_csv('data_train.csv', index_col="id")

# label encoding for target
le = preprocessing.LabelEncoder()

le.fit(data['diagnosis'])
y = le.transform(data['diagnosis'])

le.fit(data_test['diagnosis'])
y_test = le.transform(data_test['diagnosis'])

# train data
X = data.copy()
X.drop(['diagnosis'], axis=1, inplace=True)

# test data
X_test = data.copy()
X_test.drop(['diagnosis'], axis=1, inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

model = XGBClassifier(n_estimators=50, learning_rate=0.3, max_depth=2,
                      n_jobs=6, random_state=0)

scores = cross_val_score(model, X_train, y_train, cv=5)
print(scores)

model.fit(X_train, y_train,
          eval_set=[(X_valid, y_valid)],
          verbose=False)

predictions = model.predict(X_valid)
predictions_test = model.predict(X_test)

acc = accuracy_score(y_valid, predictions)
acc_test = accuracy_score(y_test, predictions_test)

print(f"Accuracy: {acc}")
print(f"Accuracy [test]: {acc}")

