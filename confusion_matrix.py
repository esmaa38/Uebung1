import fasttext
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Lade dein trainiertes Modell
model = fasttext.load_model("model_two.ftz")

# Listen für echte und vorhergesagte Labels
true_labels = []
predicted_labels = []

# Testdaten einlesen
with open("test.txt", "r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue
        label, text = line.strip().split(" ", 1)
        true_label = label.replace("__label__", "")
        prediction = model.predict(text, k=1)[0][0].replace("__label__", "")
        true_labels.append(true_label)
        predicted_labels.append(prediction)

# Confusion-Matrix erzeugen
labels = ["Negative", "Neutral", "Positive"]
cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

# Matrix anzeigen
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion-Matrix für Testdaten")
plt.savefig("confusion_matrix.png")
plt.show()





