import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv(r"C:\Users\mishr\OneDrive\Documents\Task_Management_System\learning\synthetic_tasks.csv")
print(df)

def clean_text(text):

    text = text.lower() #lowercasing

    # Remove punctuation

    text = text.translate(str.maketrans('','',string.punctuation))

    return text

df['cleaned_text'] = df['task_text'].apply(clean_text)
print(df[['task_text','cleaned_text']].head())

# Tokenization

df['tokens'] = df['cleaned_text'].apply(word_tokenize)
print(df['tokens'].head())

# Stopword Removal

stop_words = set(stopwords.words('english'))

df['token_no_stopwords'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

print(df['token_no_stopwords'].head())

#Stemming

stemmer = PorterStemmer()

df['stemmed_tokens'] = df['token_no_stopwords'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

print(df['stemmed_tokens'].head())

df.to_csv("cleaned_tasks.csv", index = False)





