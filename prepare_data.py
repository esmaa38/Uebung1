import pandas as pd
from sklearn.model_selection import train_test_split

# Lade die CSV-Datei
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']

# Wandle Zahlen in Text-Labels um
label_map = {0: "Negative", 2: "Neutral", 4: "Positive"}
df["label"] = df["polarity"].map(label_map)

# fastText-Format: __label__<Label> <Tweet>
df["formatted"] = "__label__" + df["label"] + " " + df["text"]

# Aufteilen in Train (70%), Dev (15%), Test (15%)
train, temp = train_test_split(df["formatted"], test_size=0.3, random_state=42)
dev, test = train_test_split(temp, test_size=0.5, random_state=42)

# Speichern als Textdateien
train.to_csv("train.txt", index=False, header=False)
dev.to_csv("dev.txt", index=False, header=False)
test.to_csv("test.txt", index=False, header=False)

print("Die Dateien wurden erstellt.")