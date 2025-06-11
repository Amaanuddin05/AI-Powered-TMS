import pandas as pd

df = pd.read_csv(r"C:\Users\mishr\OneDrive\Documents\Task_Management_System\learning\cleaned_tasks.csv")

print(df.head())

print("Columns:", df.columns.tolist())

#convert stemmed_tokens to numerical features usable by ML models
#TF-IDF works on strings, we first need to join the tokens into a single string

from sklearn.feature_extraction.text import TfidfVectorizer

#join tokens back to sentecne

df['processed_text'] = df['stemmed_tokens'].apply(eval).apply(lambda tokens: ' '.join(tokens))

#Initialise TF-IDF
print(df['processed_text'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

print("TF-IDF shape:", X.shape)

#Encoding Labels (category): converting task categories into numbers

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category'])

# Map: number -> category
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label map:", label_map)

#Train/Test Split: 80% for training and 20% for testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

#Train Classifiers: train two models and then compare them

#1. Naive Bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

#Predict and evaluate

nb_preds = nb_model.predict(X_test)
print("Naive Bayes Performance:\n")
print(classification_report(y_test, nb_preds, target_names=label_encoder.classes_))

#2. SVM Classifier
from sklearn.svm import LinearSVC

svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# Predict and evaluate
svm_preds = svm_model.predict(X_test)
print("SVM Performance:\n")
print(classification_report(y_test, svm_preds, target_names=label_encoder.classes_))

#Step 7: Pipeline + GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

#Building a pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

#Define GridSearch Parameters

param_grid = {
    'tfidf__max_df': [0.8, 1.0],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1, 10]  # regularization strength for SVM
}

#Running GridSearchCv
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit on your processed text and category
grid.fit(df['processed_text'], y)

#Evaluating Best Model
print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

# Predict and evaluate on test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], y, test_size=0.2, random_state=42)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))







