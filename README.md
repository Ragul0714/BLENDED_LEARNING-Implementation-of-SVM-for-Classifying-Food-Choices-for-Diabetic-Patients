# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load Data Import and prepare the dataset to initiate the analysis workflow.
   
2.Explore Data Examine the data to understand key patterns, distributions, and feature relationships.

3.Select Features Choose the most impactful features to improve model accuracy and reduce complexity.

4.Split Data Partition the dataset into training and testing sets for validation purposes.

5.Scale Features Normalize feature values to maintain consistent scales, ensuring stability during training.

6.Train Model with Hyperparameter Tuning Fit the model to the training data while adjusting hyperparameters to enhance performance.

7.Evaluate Model Assess the model’s accuracy and effectiveness on the testing set using performance metrics.


## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: RAGUL.K
RegisterNumber:  212224040258
*/

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('food_items_binary (1).csv')

print(data.head())
print(data.columns)

features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class' 

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC()
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']    
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Name:Ragul.K")
print("Register Number:212224040258")
print("Best Parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Name:Ragul.K")
print("Register Number:212224040258")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

## Output:

<img width="1090" height="748" alt="image" src="https://github.com/user-attachments/assets/c2ff2268-2e98-40be-81de-14102755bf71" />


<img width="276" height="226" alt="image" src="https://github.com/user-attachments/assets/e839f859-d193-4dd1-ad9d-2bbefd43b60a" />


<img width="665" height="91" alt="image" src="https://github.com/user-attachments/assets/e7039cb8-bf88-4211-9c7e-65477bf5c3fc" />


<img width="620" height="307" alt="image" src="https://github.com/user-attachments/assets/a9db056a-55dc-4ec6-94ec-f7fd3efc6036" />


<img width="897" height="623" alt="image" src="https://github.com/user-attachments/assets/57dc2bd3-da5f-42ba-87b8-be9a17c19607" />

## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
