import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# creating paths
Data_path = "data/raw/telco_churn.csv"
Model_Path = "models/baseline_rf.pkl"

# load the data
df = pd.read_csv(Data_path)
df.drop("customerID", axis=1, inplace = True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = 'coerce')
df.fillna(0, inplace=True)

# encode categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# feature and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# splitting the data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# model trainig
model = RandomForestClassifier(n_estimators = 100, random_state = 41)
model.fit(X_train, y_train)

# evalation
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Validation Accuracy: {acc:.4f}")
print(classification_report(y_test, preds))


# saving the model
os.makedirs(os.path.dirname(Model_Path), exist_ok=True)
joblib.dump(model, Model_Path)
print(f"Model saved at {Model_Path}")
