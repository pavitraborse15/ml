import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)

plt.subplot(3, 1, 1)
plt.plot(x, sigmoid)
plt.title("Sigmoid")

plt.subplot(3, 1, 2)
plt.plot(x, tanh)
plt.title("Tanh")

plt.subplot(3, 1, 3)
plt.plot(x, relu)
plt.title("ReLU")

plt.tight_layout()
plt.show()