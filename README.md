# Linear regression from scratch on Student Dataset

This project aims to predict student performance based on various study-related features. The project explores data preprocessing, exploratory data analysis (EDA), feature selection, and implements a custom linear regression model to predict the target variable: `Performance`.

## Table of Contents
- [Project Description](#project-description)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Custom Linear Regression Model](#custom-linear-regression-model)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualization](#visualization)
- [Installation](#installation)
- [Usage](#usage)

---

## Project Description

The objective is to analyze student data to find significant predictors of `Performance` and implement a custom linear regression model for prediction. The dataset, `student_data.csv`, contains various columns such as `Hours Studied`, `Sample Question Papers Practiced`, and others related to students' academic and study behaviors.

## Data Exploration

Exploratory data analysis is conducted using `pandas`, `matplotlib`, and `seaborn`:
- Initial inspection of the dataset using `df.head()`, `df.tail()`, and `df.info()` to understand data structure.
- Checking for missing values and examining summary statistics with `df.describe()` and visualization using heatmaps.

## Data Preprocessing

The data preprocessing steps include:
1. **Dropping irrelevant columns**: Columns like `Extracurricular Activities` are removed as they are not relevant to predicting `Performance`.
2. **Handling missing values**: Identifying columns with missing values and applying appropriate imputation techniques.
3. **Standardization**: Feature scaling of independent variables to normalize data.

## Linear Regression Model from scratch

A custom linear regression model, `LinearRegressionCustom`, is implemented in Python. Key functionalities include:
- **Fit Method**: Calculates gradients and updates weights and bias using gradient descent.
- **Predict Method**: Predicts target values based on input features.
- **Score Method**: Calculates R-squared to evaluate model performance.

### Hyperparameters:
- **Learning Rate**: 0.01
- **Number of Iterations**: 1000

## Evaluation Metrics

Several evaluation metrics are used to assess model performance:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared Score**: Measures the proportion of variance in the dependent variable that is predictable from the independent variables.

## Visualization

Visualizations are used throughout the project to understand the data and model performance:
- **Scatterplots**: Scatterplots of features vs. target variable `Performance`.
- **Loss Curve**: Plot showing the reduction of loss over epochs during training.
- **Prediction vs. Actual Scatterplot**: Visualizing predictions against actual values to evaluate model accuracy.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/student-performance-prediction.git
    ```
2. Install the required libraries:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

## Usage

1. Ensure the `student_data.csv` file is in the project directory.
2. Run the script to start the analysis and model training:
    ```bash
    python main.py
    ```

## Sample Code Snippets

To train the custom linear regression model:
```python
model = LinearRegressionCustom(learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Custom Linear Regression Model Score: {score}")
