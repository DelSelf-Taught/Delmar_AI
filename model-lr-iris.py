#import library
import numpy as np
from sklearn.datasets import load_iris                           # data used
from sklearn.metrics import r2_score, mean_squared_error         # User for avaliable model  
import matplotlib.pyplot as plt                                  # Used for plot
from sklearn.linear_model import LinearRegression                # Model
from sklearn.decomposition import PCA                            # For Reducion Dimensionality
from sklearn.model_selection import train_test_split             # Train/Test model
from sklearn.pipeline import Pipeline as pl                      # Pipeline Organization


# Data
iris = load_iris()

# Data passing X and Y
x = iris.data
y = iris.target

# Train and test
x_tr, x_ts, y_tr, y_ts =  train_test_split(x,y, test_size=0.3)

# Model
model = LinearRegression().fit(x,y)

# Model prediction on the x-axis
y_pred = model.predict(x_tr)

# Plot
rgb = np.array(['r', 'g', 'b'])
plt.scatter(y_pred, x_tr[:, 0], c=rgb[y_tr]) 
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Scatter Plot of Iris Data Predictions vs. Actual (Training Data)")
plt.legend(iris.target_names)
