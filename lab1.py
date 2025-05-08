
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2)

def run(opt, name):
    print(f"\nTraining with activation=tanh and optimizer={name}")
    m = Sequential([Input(shape=(4,)), Dense(10, activation='tanh'), Dense(3, activation='softmax')])
    m.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
    m.fit(X_train, y_train, epochs=20, verbose=0)
    acc = m.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Test Accuracy: {acc:.4f}")
    return acc

opts = [('SGD', SGD(0.01)), ('Adam', Adam(0.01)), ('RMSprop', RMSprop(0.01))]
results = [(n, run(o, n)) for n, o in opts]

plt.bar(*zip(*results), color=['b', 'g', 'r'])
plt.title("Optimizers vs Accuracy"); plt.ylim(0, 1); plt.ylabel("Accuracy"); plt.show()