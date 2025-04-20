import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("best_model.h5")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

preds = model.predict(X_test)
preds = np.argmax(preds, axis=-1)

for i in range(3):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[i])
    plt.title("Input")
    plt.subplot(1, 3, 2)
    plt.imshow(y_test[i])
    plt.title("Ground Truth")
    plt.subplot(1, 3, 3)
    plt.imshow(preds[i])
    plt.title("Prediction")
    plt.show()