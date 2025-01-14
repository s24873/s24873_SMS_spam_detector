import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../ml_data/spam.csv", encoding="ISO-8859-1")
data = data[['v1', 'v2']].copy()
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

data['message_length'] = data['message'].apply(len)
data['num_digits'] = data['message'].apply(lambda x: sum(c.isdigit() for c in x))
data['num_uppercase'] = data['message'].apply(lambda x: sum(c.isupper() for c in x))
data['num_special_chars'] = data['message'].apply(lambda x: sum(not c.isalnum() for c in x))
data['num_words'] = data['message'].apply(lambda x: len(x.split()))
data['avg_word_length'] = data['message'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if len(x.split()) > 0 else 0)
data['num_exclamations'] = data['message'].apply(lambda x: x.count('!'))
data['num_questions'] = data['message'].apply(lambda x: x.count('?'))

X = data.drop(columns=['label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)

