import fasttext

# Trainiere das Modell
model = fasttext.train_supervised(input="train.txt")

# Speichern
model.save_model("model_default.ftz")

# Testen auf Development-Daten
n, precision, recall = model.test("dev.txt")

# F1-Score berechnen
f1 = 2 * (precision * recall) / (precision + recall)

# Ausgabe
print(f"Samples: {n}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

