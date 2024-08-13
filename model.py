
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pickle


df = pd.read_csv('train_loan_preprocess.csv')
df = df.drop('Loan_ID', axis=1)


imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


le = LabelEncoder()
for column in df_imputed.select_dtypes(include=['object']).columns:
    df_imputed[column] = le.fit_transform(df_imputed[column])


X = df_imputed.drop('Loan_Status', axis=1)
y = df_imputed['Loan_Status']


print("Feature names from training:", X.columns.tolist())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


with open('loan_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('features.pkl', 'wb') as features_file:
    pickle.dump(X.columns.tolist(), features_file)

print("Model saved as 'loan_model.pkl'")
print("Features saved as 'features.pkl'")
