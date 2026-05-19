# SGD-Classifier :

## Developed by: Yuvan shankar M
## RegisterNumber:  212224220126

## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the dataset and required libraries
   Load the Iris dataset and import necessary Python libraries for data processing and model building.

2. Split and preprocess the data
   Divide the dataset into training and testing sets and standardize the feature values.

3. Initialize and train the SGD Classifier
   Create an SGD Classifier model and train it using the training data.

4. Predict and evaluate the model
   Predict the Iris flower species using test data and measure accuracy of the model.


## Program:

Program to implement the prediction of iris species using SGD Classifier.



```py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Feature scaling (important for SGD)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SGD Classifier
model = SGDClassifier(max_iter=1000, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

```

## Output:

<img width="596" height="247" alt="image" src="https://github.com/user-attachments/assets/38c7b60e-6100-4e50-a8af-a6359dd497ec" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
