# Advanced ML Model Training Web Application

A sophisticated web application for training and evaluating ensemble machine learning models with comprehensive data analysis and visualization capabilities.

## Features

### Data Handling
- Upload CSV and Excel files
- Pre-loaded example datasets:
  - Iris (Classification)
  - California Housing (Regression)
  - Diabetes (Regression)
  - Breast Cancer (Classification)
  - Wine (Classification)
- Automatic data type detection and preprocessing
- Missing value handling
- Feature scaling

### Data Analysis
- **Comprehensive EDA**:
  - Basic Statistics
  - Distribution Analysis
  - Correlation Matrix
  - Feature Relationships
  - Outlier Analysis
  - Missing Value Analysis
  - Categorical Data Analysis

- **Bias Analysis**:
  - Class Distribution
  - Feature Skewness
  - Imbalance Detection
  - Feature Scaling Recommendations

### Model Training
- **Ensemble Methods**:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost

- **Automatic Problem Type Detection**:
  - Classification
  - Regression
  - User Confirmation

- **Model Training Features**:
  - Customizable Train-Test Split
  - Class Imbalance Handling
  - Cross-Validation
  - Feature Importance Analysis

### Hyperparameter Tuning
- Optuna-based Optimization
- Multiple Trials Support
- Best Parameter Selection
- Performance Tracking

### Model Evaluation
- **Classification Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

- **Regression Metrics**:
  - R-squared
  - MSE
  - RMSE
  - MAE

### Visualizations
- Interactive Plotly Graphs
- Feature Importance Plots
- SHAP Value Analysis
- Learning Curves
- Actual vs Predicted Plots

### Model Persistence
- Save Trained Models
- Export Visualizations
- Download Analysis Reports

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`
3. Follow the intuitive step-by-step process:
   - Upload data or select example dataset
   - Review data analysis and bias report
   - Confirm problem type
   - Select target column and set parameters
   - Choose ensemble model
   - Train and evaluate
   - Optionally tune hyperparameters
   - Save model and results

## Error Handling

The application includes comprehensive error handling for:
- Invalid file formats
- Missing data
- Training failures
- Memory constraints
- Browser compatibility issues

## Requirements

See `requirements.txt` for a complete list of dependencies. Key packages include:
- Flask 2.0+
- Pandas 1.3+
- NumPy 1.21+
- Scikit-learn 1.0+
- XGBoost 1.5+
- LightGBM 3.3+
- CatBoost 1.0+
- Plotly 5.3+
- Optuna 2.10+
- SHAP 0.40+
