import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def visualize_predictions(model_path, X_test_path, y_test_path, num_samples=5):
    model = load_model(model_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    preds = model.predict(X_test)
    preds = np.argmax(preds, axis=-1)

    for i in range(num_samples):
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[i])
        plt.title("Input")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(y_test[i])
        plt.title("Ground Truth")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(preds[i])
        plt.title("Prediction")
        plt.axis('off')
        plt.tight_layout()
        plt.show()