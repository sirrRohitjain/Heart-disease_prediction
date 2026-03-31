import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

model = load_model('heart_disease_model.h5')
model.summary()
for layer in model.layers:
    w = layer.get_weights()
    if w:  # only if the layer has weights
        weights, biases = w
        print(f"Layer: {layer.name}")
        print("Weights shape:", weights.shape)
        print("Biases shape:", biases.shape)
        print("="*50)
    else:
        print(f"Layer: {layer.name} has no weights.")
        print("="*50)

# Example patient data
print("="*50)
new_data = pd.DataFrame({
    'age': [63],
    'trestbps': [145],
    'chol': [233],
    'thalach': [150],
    'oldpeak': [2.3],
    'slope': [3],
    'sex_0': [0], 'sex_1': [1],      # one-hot encoded categorical columns
    'cp_0': [0], 'cp_1': [1], 'cp_2':[0], 'cp_3':[0],
    'fbs_0': [1], 'fbs_1':[0],
    'restecg_0':[1], 'restecg_1':[0], 'restecg_2':[0],
    'exang_0':[0], 'exang_1':[1],
    'ca_0':[0], 'ca_1':[1], 'ca_2':[0], 'ca_3':[0], 'ca_4':[0],
    'thal_0':[0], 'thal_1':[1], 'thal_2':[0], 'thal_3':[0]
})
numeric_cols = ['age','trestbps','chol','thalach','oldpeak','slope']
means = np.array([63, 145, 233, 150, 2.3, 3])  # replace with your training means
stds = np.array([9.43, 17.54, 51.97, 22.95, 1.29, 0.61])  # replace with your training stds

new_data[numeric_cols] = (new_data[numeric_cols] - means) / stds
new_x = new_data.to_numpy(dtype=np.float32)
# Predict probability of heart disease
pred_probs = model.predict(new_x)
print("Predicted probability:", pred_probs)

# Convert probability to binary class
pred_class = (pred_probs > 0.5).astype(int)
print("Predicted class:", pred_class)