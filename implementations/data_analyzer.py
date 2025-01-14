import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv

# poki co odczytywanie z pliku z katalogu projektu ze wzgledu na bledy z dostepem po pobraniu datasetu przez kagglehub
data_path = "../ml_data/spam.csv"
data = pd.read_csv(data_path, encoding="ISO-8859-1")

print("\nPierwsze 5 wierszy:")
print(data.head())

data.columns = ['label', 'message']

print("\nRozklad klas:")
print(data['label'].value_counts())

print("\nPodstawowe statystyki opisowe:")
print(data.describe())

print("\nBrakujace wartosci:")
print(data.isnull().sum())

# wizualizacja rozkladu zmiennych
plt.figure(figsize=(10, 5))
sns.countplot(data=data, x='label')
plt.title('Rozklad spam vs ham')
plt.show()
report = sv.analyze(data)
report.show_html("Spam_Dataset_Report.html")

# histogram dlugosci wiadomosci
data['message_length'] = data['message'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='message_length', hue='label', bins=50, kde=True)
plt.title('Rozklad dlugosci wiadomosci(spam vs ham)')
plt.xlabel('Dlugosc wiadomosci')
plt.show()



