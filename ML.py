import numpy as np
from sklearn.linear_model import LinearRegression

# Create a training dataset
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([10, 20, 30])

# Create a linear regression model
model = LinearRegression()

# Train the model on the training dataset
model.fit(X, y)

# Make a prediction
new_input = np.array([7, 8])
prediction = model.predict(new_input)

# Print the prediction
print(prediction)
