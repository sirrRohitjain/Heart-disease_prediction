import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# For reproducibility
keras.utils.set_random_seed(42)

# Load dataset
df = pd.read_csv("dataset.csv")

# Identify categorical and numeric columns
categorical_variables = ['sex', 'cp', 'fbs', 'restecg','exang', 'ca', 'thal']
numerics = ['age', 'trestbps','chol', 'thalach', 'oldpeak', 'slope']

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=categorical_variables)

# Split into train/test
test_df = df.sample(frac=0.2, random_state=42)
train_df = df.drop(test_df.index)

# Convert numeric columns to float to avoid int64 errors
train_df[numerics] = train_df[numerics].astype('float64')
test_df[numerics] = test_df[numerics].astype('float64')

# Normalize numeric columns
means = train_df[numerics].mean()
stds = train_df[numerics].std()
train_df.loc[:, numerics] = (train_df[numerics] - means) / stds
test_df.loc[:, numerics] = (test_df[numerics] - means) / stds

# Split into features and target
train_x = train_df.drop(columns=['target']).to_numpy(dtype=np.float32)
train_y = train_df['target'].to_numpy(dtype=np.float32).reshape(-1,1)
test_x = test_df.drop(columns=['target']).to_numpy(dtype=np.float32)
test_y = test_df['target'].to_numpy(dtype=np.float32).reshape(-1,1)

# Neural network model
num_cols = train_x.shape[1]
input_layer = Input(shape=(num_cols,))
h = Dense(64, activation='relu')(input_layer)
output_layer = Dense(1, activation='sigmoid')(h)
model = Model(inputs=input_layer, outputs=output_layer)

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_x, train_y,
    epochs=100,
    batch_size=32,
    verbose=1,
    validation_data=(test_x, test_y)
)

# Save model
model.save('heart_disease_model.h5')
print("Model saved successfully!")
model.summary()
# Example: load model and predict
# from tensorflow.keras.models import load_model
# model = load_model('heart_disease_model.h5')
# predictions = model.predict(test_x)
# predicted_classes = (predictions > 0.5).astype(int)
# print(predicted_classes[:10])