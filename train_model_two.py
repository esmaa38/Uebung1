import fasttext

# 1. Modell trainieren mit Parametern
model = fasttext.train_supervised(
    input="train.txt",
    lr=0.5,
    epoch=25,
    wordNgrams=2,
    dim=100
)

# 2. Modell speichern
model.save_model("model_two.ftz")

# 3. Testen auf dem Development-Datensatz
n_dev, precision_dev, recall_dev = model.test("dev.txt")
f1_dev = 2 * (precision_dev * recall_dev) / (precision_dev + recall_dev)

print("== Ergebnisse auf dem Development-Datensatz ==")
print(f"Samples: {n_dev}")
print(f"Precision: {precision_dev:.4f}")
print(f"Recall: {recall_dev:.4f}")
print(f"F1-Score: {f1_dev:.4f}")

# 4. Testen auf dem Test-Datensatz  <<< DAS IST SCHRITT 6
n_test, precision_test, recall_test = model.test("test.txt")
f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)

print("\n== Ergebnisse auf dem Test-Datensatz ==")
print(f"Samples: {n_test}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1-Score: {f1_test:.4f}")
