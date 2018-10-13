import numpy as np
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")
X = mnist.data/255.0
y = mnist.target

print(X.shape, y.shape)

