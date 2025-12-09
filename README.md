# Linear Regression Supervised Learning 📊

A Python-based implementation of **supervised linear regression** from scratch, demonstrating the fundamental concepts of machine learning model training and evaluation.

## Overview 🎯

This project implements a simple linear regression model using the mathematical formula:

$$y = b_0 + b_1 \cdot x$$

The model is trained on a salary dataset to predict employee salaries based on years of experience.

## Features ✨

- **Data Loading & Processing** 📁
  - Reads CSV data from `Salary_dataset.csv`
  - Parses and structures data into training features and labels

- **Model Training** 🧠
  - Calculates optimal coefficients (b₀ and b₁) using least squares method
  - Numerator: $\sum (x_i - \bar{x})(y_i - \bar{y})$
  - Denominator: $\sum (x_i - \bar{x})^2$

- **Predictions** 🔮
  - Makes salary predictions for new experience values
  - Uses the trained model on test data

- **Evaluation Metrics** 📈
  - **Mean Absolute Error (MAE)** - Average absolute prediction error
  - **Mean Squared Error (MSE)** - Average squared prediction error
  - **Root Mean Squared Error (RMSE)** - Square root of MSE
  - **R² Score** (Goodness of Fit) - Model performance metric (0-1 scale)

- **Visualization** 📉
  - Plots regression line against actual data points
  - Uses Matplotlib for clear visualization

## Dataset 📊

- **Source**: `Salary_dataset.csv`
- **Features**: Years of Experience
- **Target**: Salary
- **Records**: 30 employees
- **Experience Range**: 1.2 - 10.6 years
- **Salary Range**: $37,732 - $122,392

## Model Performance 🎯

- **MAE**: ~4,015.28
- **MSE**: ~21,521,064.17
- **R² Score**: ~0.989 (Excellent fit!)

## Usage 🚀

Run the Jupyter notebook to:

1. Load and explore the salary dataset
2. Extract features (experience) and target (salary)
3. Calculate mean values
4. Train the linear regression model
5. Generate predictions on test data
6. Compute error metrics
7. Visualize results

## Mathematical Concepts 📐

### Coefficient Calculation

$$b_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$

$$b_0 = \bar{y} - b_1 \cdot \bar{x}$$

### Performance Metrics

$$MAE = \frac{1}{n}\sum |y_{pred} - y_{actual}|$$

$$MSE = \frac{1}{n}\sum (y_{pred} - y_{actual})^2$$

$$R^2 = 1 - \frac{MSE_{residual}}{MSE_{total}}$$

## Technologies Used 💻

- **Python 3.x**
- **Jupyter Notebook** - Interactive development environment
- **Matplotlib** - Data visualization
- **NumPy** - Numerical computations

## Files 📄

- `linear_regression_supervised.ipynb` - Main notebook with complete implementation

## Key Insights 💡

- Strong positive correlation between years of experience and salary
- R² score of 0.989 indicates the model explains 98.9% of variance
- Linear model is appropriate for this dataset
- Model generalizes well on test data

## Future Enhancements 🔮

- Implement multiple linear regression (multiple features)
- Add polynomial regression
- Use scikit-learn for comparison
- Cross-validation for better evaluation
- Residual analysis

# Screenshots 📸

<img width="905" height="637" alt="Screenshot 2025-12-09 224131" src="https://github.com/user-attachments/assets/19a68049-0e24-4a73-84b5-6c83671cca7a" />

---

**Author**: Naviy  
**Status**: ✅ Complete  
**Last Updated**: 2025
