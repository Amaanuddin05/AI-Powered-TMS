import pandas as pd

#load preprocessed file from Week 1

df = pd.read_csv(r"C:\Users\mishr\OneDrive\Documents\Task_Management_System\learning\cleaned_tasks.csv");

print(df.head())
print("Columns:",df.columns.tolist())

from datetime import datetime

#Feature 1: Text length

df['text_length'] = df['task_text'].apply(lambda x: len(x.split()))

# Feature 2: Days left until deadline
df['due_date'] = pd.to_datetime(df['due_date']) #convert string to date
today = datetime.today()

df['due_days_left'] = (df['due_date']-today).dt.days

# Keep only non-negative due dates
df = df[df['due_days_left'] >= 0]

#Show sample feature
print(df[['task_text', 'priority', 'text_length', 'category', 'due_days_left']].head())

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#Encode 'category' using One-Hot

ohe = OneHotEncoder(sparse_output=False)
category_encoded = ohe.fit_transform(df[['category']])
category_cols = ohe.get_feature_names_out(['category'])

#combine a new data frame with these features
category_df = pd.DataFrame(category_encoded, columns=category_cols, index=df.index)


# Feature 3: Urgency flag
df['urgency_flag'] = df['due_days_left'].apply(lambda x: 1 if x <= 2 else 0)

# Feature 4: One-hot encode 'status'
status_ohe = pd.get_dummies(df['status'], prefix='status')

# Combine all features together
features_df = pd.concat([
    df[['text_length', 'due_days_left', 'urgency_flag']].reset_index(drop=True),
    category_df.reset_index(drop=True),
    status_ohe.reset_index(drop=True)
], axis=1)


#Encode priority label
le = LabelEncoder()
df['priority_binary'] = df['priority'].apply(lambda x: 1 if x == 'High' else 0)

#Final features and target
X = features_df
y = df['priority_binary'].reset_index(drop=True) 


#split train test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("Random Forest Performance:\n")
print(classification_report(y_test, rf_preds, target_names=["Not High", "High"]))


from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

print("XGBoost Performance:\n")
print(classification_report(y_test, rf_preds, target_names=["Not High", "High"]))


import numpy as np

users = ['Alice', 'Bob', 'Charlie', 'Diana']
df['assigned_to'] = np.random.choice(users, size=len(df))

df['predicted_priority'] = rf.predict(X)

# Filter high-priority predicted tasks
high_tasks = df[df['predicted_priority'] == 1]

# Count how many high tasks per user
user_load = high_tasks['assigned_to'].value_counts()
print("High-priority task load per user:\n")
print(user_load)

max_capacity = 5

# Flag overloaded users
overloaded_users = user_load[user_load > max_capacity].index.tolist()
print("\nOverloaded users:", overloaded_users)

# Get tasks that should be reassigned
reassign_df = high_tasks[high_tasks['assigned_to'].isin(overloaded_users)]

# Reassign randomly to users under capacity
for idx, row in reassign_df.iterrows():
    # Find a user under capacity
    for user in users:
        if user_load.get(user, 0) < max_capacity:
            df.at[idx, 'assigned_to'] = user
            user_load[user] = user_load.get(user, 0) + 1
            break


df.to_csv("balanced_tasks.csv", index=False)

import joblib
joblib.dump((rf, features_df.columns.tolist()), "week3_rf_model.pkl")




