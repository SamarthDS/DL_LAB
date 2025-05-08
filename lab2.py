# 2nd Program

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal, GlorotUniform, RandomNormal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# Data preparation
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Store results
results = {}

# Train function
def train(init):
    model = Sequential([
        Dense(10, activation='relu', kernel_initializer=init, input_shape=(4,)),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, verbose=0)
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    name = type(init).__name__
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Evaluate all initializers
for initializer in [HeNormal(), GlorotUniform(), RandomNormal(0., 0.05)]:
    train(initializer)

# Plotting the comparison
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Comparison of Weight Initializers')
plt.ylabel('Test Accuracy')
plt.ylim(0.0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
