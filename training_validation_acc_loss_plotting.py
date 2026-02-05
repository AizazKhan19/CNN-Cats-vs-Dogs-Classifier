import json
import matplotlib.pyplot as plt
import os

# Path to the history.json file
history_file = os.path.join('Training History', 'training_history.json')

# Load the history JSON
with open(history_file, 'r') as f:
    history = json.load(f)

# -------------------------------
# Plot Training and Validation Accuracy
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Plot Training and Validation Loss
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(history['loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
