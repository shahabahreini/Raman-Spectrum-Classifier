import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

# Simulate some data
signal = np.sin(np.linspace(0, 20 * np.pi, 1000)) + 0.5 * np.random.randn(1000)

# Prepare data (this is just a mockup, in a real scenario you'd split your signal into segments and label them)
X = np.array([signal[i:i+10] for i in range(0, len(signal)-10)])
y = np.zeros((X.shape[0], 3))  # Three classes: peak, valley, none
# Here, you'd set the labels based on your segmented data

# Create a simple CNN model
model = Sequential([
    Conv1D(16, 3, activation='relu', input_shape=(10, 1)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # Three classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Predict on new signals
predictions = model.predict(X)

# For each segment, you can extract the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)
