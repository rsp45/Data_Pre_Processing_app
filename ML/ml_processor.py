import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                            GradientBoostingClassifier, GradientBoostingRegressor,
                            ExtraTreesClassifier, ExtraTreesRegressor,
                            AdaBoostClassifier, AdaBoostRegressor,
                            BaggingClassifier, BaggingRegressor,
                            VotingClassifier, VotingRegressor,
                            StackingClassifier, StackingRegressor)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna
import shap
import plotly.graph_objects as go
import plotly.express as px
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os
from datetime import datetime
import traceback

# Model aliases for better user experience
MODEL_ALIASES = {
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
    'lgb': 'LightGBM',
    'cat': 'CatBoost',
    'lr': 'Linear Regression',
    'lasso': 'Lasso Regression',
    'ridge': 'Ridge Regression',
    'svm': 'Support Vector Machine',
    'dt': 'Decision Tree',
    'knn': 'K-Nearest Neighbors'
}

# Model aliases for more intuitive selection
MODEL_ALIASES.update({
    'gb': 'gradient_boosting',
    'et': 'extra_trees',
    'ada': 'adaboost',
    'bag': 'bagging',
    'vote': 'voting',
    'stack': 'stacking'
})

class MLProcessor:
    def __init__(self, data_path=None, data=None):
        """Initialize MLProcessor with either a data path or pandas DataFrame"""
        self.data = None
        self.target = None
        self.problem_type = None
        self.is_classification = None
        self.models = {}
        self.current_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None  
        self.logger = logging.getLogger(__name__)

        try:
            if data_path:
                self.load_data(data_path)
            elif isinstance(data, pd.DataFrame):
                self.data = data.copy()
            else:
                raise ValueError("Either data_path or data must be provided")
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    def load_data(self, data_path):
        """Load data from file"""
        try:
            if data_path.endswith('.csv'):
                self.data = pd.read_csv(data_path)
            elif data_path.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(data_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            self.logger.info(f"Successfully loaded data from {data_path}")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def set_target(self, target_column):
        """Set the target column and determine the problem type."""
        try:
            if target_column not in self.data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            self.target = target_column
            target_data = self.data[target_column]
            
            # Determine problem type
            unique_values = target_data.nunique()
            if pd.api.types.is_numeric_dtype(target_data):
                if unique_values <= 10 and all(target_data.dropna() == target_data.dropna().astype(int)):
                    self.problem_type = 'classification'
                    self.is_classification = True
                else:
                    self.problem_type = 'regression'
                    self.is_classification = False
            else:
                self.problem_type = 'classification'
                self.is_classification = True
            
            self.logger.info(f"Target set to '{target_column}'. Problem type: {self.problem_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting target: {str(e)}")
            raise

    def analyze_data_bias(self):
        """Analyze data bias and class imbalance"""
        try:
            if self.X is None or self.y is None:
                raise ValueError("Please call detect_problem_type first to set X and y")
                
            bias_report = {}
            
            # Check class distribution for classification
            if self.problem_type == 'classification':
                class_dist = self.y.value_counts(normalize=True)
                # Convert numpy types to Python native types
                bias_report['class_distribution'] = {
                    str(k): float(v) for k, v in class_dist.to_dict().items()
                }
                
                # Calculate class imbalance ratio
                max_class = float(class_dist.max())
                min_class = float(class_dist.min())
                bias_report['imbalance_ratio'] = max_class / min_class
                
                # Suggest if SMOTE is needed (convert numpy.bool_ to Python bool)
                bias_report['needs_smote'] = bool(bias_report['imbalance_ratio'] > 3)
            
            # Check feature distributions and skewness
            numerical_features = self.X.select_dtypes(include=['int64', 'float64']).columns
            feature_stats = {}
            
            for col in numerical_features:
                col_stats = self.X[col].describe()
                skewness = float(self.X[col].skew())
                mean = float(col_stats['mean'])
                std = float(col_stats['std'])
                needs_scaling = bool(abs(mean) > 1 or std > 1)
                
                feature_stats[str(col)] = {
                    'skewness': skewness,
                    'mean': mean,
                    'std': std,
                    'needs_scaling': needs_scaling
                }
            
            bias_report['feature_statistics'] = feature_stats
            
            return bias_report
            
        except Exception as e:
            self.logger.error(f"Error in bias analysis: {str(e)}")
            raise

    def detect_problem_type(self, target_column):
        """Detect if this is a classification or regression problem"""
        try:
            self.target_column = target_column
            self.y = self.data[target_column]
            self.X = self.data.drop(columns=[target_column])
            
            # Check if target is categorical or numerical
            unique_count = self.y.nunique()
            if pd.api.types.is_numeric_dtype(self.y):
                if unique_count < 10 and all(self.y.astype(int) == self.y):
                    self.problem_type = 'classification'
                else:
                    self.problem_type = 'regression'
            else:
                self.problem_type = 'classification'
                
            return self.problem_type
            
        except Exception as e:
            self.logger.error(f"Error in problem type detection: {str(e)}")
            raise

    def perform_eda(self):
        """Perform exploratory data analysis"""
        try:
            if self.data is None:
                raise ValueError("No data loaded. Please upload data first.")
                
            if self.target_column is None:
                raise ValueError("Target column not set. Please set target column first.")

            # Basic data summary
            summary = {
                'total_samples': len(self.data),
                'num_features': len(self.data.columns),
                'missing_values': self.data.isnull().sum().sum()
            }

            # Target analysis
            target_data = self.data[self.target_column]
            target_analysis = {
                'type': str(target_data.dtype),
                'unique_values': target_data.nunique()
            }

            if target_data.dtype in ['object', 'category'] or self.problem_type == 'classification':
                class_dist = target_data.value_counts().to_dict()
                target_analysis['class_distribution'] = {str(k): int(v) for k, v in class_dist.items()}

            # Feature analysis
            feature_analysis = {}
            for column in self.data.columns:
                if column != self.target_column:
                    col_data = self.data[column]
                    analysis = {
                        'type': str(col_data.dtype),
                        'missing': int(col_data.isnull().sum()),
                        'unique': int(col_data.nunique())
                    }
                    
                    if col_data.dtype in ['int64', 'float64']:
                        analysis.update({
                            'mean': float(col_data.mean()),
                            'std': float(col_data.std()),
                            'skew': float(col_data.skew())
                        })
                    
                    feature_analysis[column] = analysis

            # Generate recommendations
            recommendations = {
                'encoding': [],
                'scaling': [],
                'imputation': [],
                'feature_selection': []
            }

            # Recommend encoding for categorical features
            cat_features = self.data.select_dtypes(include=['object', 'category']).columns
            recommendations['encoding'].extend([str(col) for col in cat_features])

            # Recommend scaling for numerical features with large values or high variance
            num_features = self.data.select_dtypes(include=['int64', 'float64']).columns
            for col in num_features:
                if col != self.target_column:
                    if abs(self.data[col].mean()) > 1 or self.data[col].std() > 1:
                        recommendations['scaling'].append(str(col))

            # Recommend imputation for features with missing values
            for col in self.data.columns:
                if col != self.target_column and self.data[col].isnull().sum() > 0:
                    recommendations['imputation'].append(str(col))

            # Recommend feature selection if there are many features
            if len(self.data.columns) > 10:
                recommendations['feature_selection'] = True

            return {
                'summary': summary,
                'target_analysis': target_analysis,
                'feature_analysis': feature_analysis,
                'recommendations': recommendations,
                'problem_type': self.problem_type
            }

        except Exception as e:
            self.logger.error(f"Error in EDA: {str(e)}")
            raise

    def preprocess_data(self, test_size=0.2, handle_imbalance=True):
        """Preprocess the data for model training."""
        try:
            if self.data is None or self.target is None:
                raise ValueError("Data or target not set")
                
            if len(self.data) == 0:
                raise ValueError("Dataset is empty")

            preprocessing_steps = []
            
            # Store original shapes for logging
            original_shape = self.data.shape
            self.logger.info(f"Original data shape: {original_shape}")
            
            # Separate features and target
            X = self.data.drop(columns=[self.target])
            y = self.data[self.target]
            
            # Store initial feature names
            initial_features = list(X.columns)
            
            # Identify column types
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            datetime_cols = X.select_dtypes(include=['datetime64']).columns
            
            # Handle datetime features
            for col in datetime_cols:
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X = X.drop(columns=[col])
                preprocessing_steps.append(f"Extracted year, month, day from {col}")
            
            # Handle missing values
            if X.isnull().sum().sum() > 0:
                # Numeric imputation
                if len(numeric_cols) > 0:
                    num_imputer = SimpleImputer(strategy='mean')
                    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
                    preprocessing_steps.append("Imputed missing numeric values with mean")
                
                # Categorical imputation
                if len(categorical_cols) > 0:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
                    preprocessing_steps.append("Imputed missing categorical values with mode")
            
            # Encode categorical variables
            encoders = {}
            for col in categorical_cols:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])
                encoders[col] = encoder
                preprocessing_steps.append(f"Encoded categorical column: {col}")
            
            # Scale numeric features
            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
                preprocessing_steps.append("Scaled numeric features")
            
            # Store final feature names
            self.feature_names = list(X.columns)
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, 
                stratify=y if self.is_classification else None
            )
            preprocessing_steps.append(f"Split data into train/test sets (test_size={test_size})")
            
            # Handle class imbalance if needed
            if handle_imbalance and self.is_classification:
                # Check class distribution
                class_dist = pd.Series(self.y_train).value_counts()
                min_samples = class_dist.min()
                if min_samples < len(self.y_train) * 0.2:  # If minority class < 20%
                    smote = SMOTE(random_state=42)
                    self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                    preprocessing_steps.append("Applied SMOTE to handle class imbalance")
            
            # Log shapes after preprocessing
            self.logger.info(f"Data shapes after preprocessing - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
            
            return {
                'status': 'success',
                'message': 'Data preprocessing completed successfully',
                'steps': preprocessing_steps,
                'shapes': {
                    'original': original_shape,
                    'train': self.X_train.shape,
                    'test': self.X_test.shape
                },
                'feature_names': self.feature_names,
                'initial_features': initial_features,
                'target_distribution': pd.Series(self.y_train).value_counts().to_dict() if self.is_classification else None
            }
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train_model(self, model_type, custom_params=None):
        """Train a model and store it for comparison."""
        try:
            self.logger.info("Starting model training...")
            
            if self.X_train is None or self.y_train is None:
                raise ValueError("Data not preprocessed. Please preprocess data first.")
            
            if not self.feature_names:
                self.feature_names = list(self.X_train.columns)
                self.logger.warning("Feature names not found, using column names from training data")
            
            self.logger.info(f"Current shapes - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
            
            # Get model instance
            self.logger.info(f"Getting model instance for type: {model_type}")
            model = self._get_model(model_type)
            
            # Set custom parameters if provided
            if custom_params:
                model.set_params(**custom_params)
            
            # Train model
            self.logger.info("Training model...")
            model.fit(self.X_train, self.y_train)
            
            # Get predictions
            train_predictions = model.predict(self.X_train)
            test_predictions = model.predict(self.X_test)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(self.y_train, train_predictions)
            test_metrics = self._calculate_metrics(self.y_test, test_predictions)
            
            # Get feature importance if available
            importance_dict = {}
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_names, model.feature_importances_))
                importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            # Store model and results
            model_name = MODEL_ALIASES.get(model_type, model_type)
            self.models[model_name] = {
                'model': model,
                'metrics': test_metrics,
                'predictions': test_predictions,
                'importance': importance_dict
            }
            
            self.logger.info(f"Model {model_name} trained and stored. Total models: {len(self.models)}")
            
            # Prepare response
            response = {
                'status': 'success',
                'message': f'Successfully trained {model_name} model',
                'model_type': model_name,
                'metrics': test_metrics,
                'feature_importance': {
                    'type': 'feature_importance',
                    'data': importance_dict
                },
                'data_shapes': {
                    'train_shape': list(self.X_train.shape),
                    'test_shape': list(self.X_test.shape)
                }
            }
            
            self.logger.info("Complete response: " + str(response))
            return response
            
        except Exception as e:
            self.logger.error(f"Error in train_model: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics based on problem type."""
        try:
            if self.problem_type == 'classification':
                return {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted'),
                    'recall': recall_score(y_true, y_pred, average='weighted'),
                    'f1': f1_score(y_true, y_pred, average='weighted')
                }
            else:  # regression
                return {
                    'r2': r2_score(y_true, y_pred),
                    'mse': mean_squared_error(y_true, y_pred),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
                }
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            raise

    def _get_model(self, model_type: str):
        """Get the model instance based on type"""
        try:
            # Convert display name to key if needed
            if model_type in MODEL_ALIASES.values():
                model_type = {v: k for k, v in MODEL_ALIASES.items()}[model_type]

            # Get base model
            if model_type == 'rf':
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42) if self.is_classification else RandomForestRegressor(n_estimators=100, random_state=42)
                
            elif model_type == 'xgb':
                import xgboost as xgb
                return xgb.XGBClassifier(random_state=42) if self.is_classification else xgb.XGBRegressor(random_state=42)
                
            elif model_type == 'lgb':
                import lightgbm as lgb
                return lgb.LGBMClassifier(random_state=42) if self.is_classification else lgb.LGBMRegressor(random_state=42)
                
            elif model_type == 'cat':
                from catboost import CatBoostClassifier, CatBoostRegressor
                return CatBoostClassifier(verbose=False, random_state=42) if self.is_classification else CatBoostRegressor(verbose=False, random_state=42)
                
            elif model_type == 'gb':
                from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
                return GradientBoostingClassifier(random_state=42) if self.is_classification else GradientBoostingRegressor(random_state=42)
                
            elif model_type == 'et':
                from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
                return ExtraTreesClassifier(random_state=42) if self.is_classification else ExtraTreesRegressor(random_state=42)
                
            elif model_type == 'ada':
                from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
                return AdaBoostClassifier(random_state=42) if self.is_classification else AdaBoostRegressor(random_state=42)
                
            elif model_type == 'bag':
                from sklearn.ensemble import BaggingClassifier, BaggingRegressor
                return BaggingClassifier(random_state=42) if self.is_classification else BaggingRegressor(random_state=42)
                
            elif model_type == 'lr':
                from sklearn.linear_model import LogisticRegression, LinearRegression
                return LogisticRegression(random_state=42) if self.is_classification else LinearRegression()
                
            elif model_type == 'lasso':
                from sklearn.linear_model import Lasso
                if self.is_classification:
                    raise ValueError("Lasso regression is not suitable for classification problems")
                return Lasso(random_state=42)
                
            elif model_type == 'ridge':
                from sklearn.linear_model import Ridge
                if self.is_classification:
                    raise ValueError("Ridge regression is not suitable for classification problems")
                return Ridge(random_state=42)
                
            elif model_type == 'svm':
                from sklearn.svm import SVC, SVR
                return SVC(random_state=42) if self.is_classification else SVR()
                
            elif model_type == 'dt':
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                return DecisionTreeClassifier(random_state=42) if self.is_classification else DecisionTreeRegressor(random_state=42)
                
            elif model_type == 'knn':
                from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
                return KNeighborsClassifier() if self.is_classification else KNeighborsRegressor()
                
            else:
                available_models = set(MODEL_ALIASES.keys()) | set(MODEL_ALIASES.values())
                raise ValueError(f"Invalid model type '{model_type}'. Available models: {sorted(available_models)}")
                
        except Exception as e:
            self.logger.error(f"Error getting model: {str(e)}")
            raise

    def get_model_comparison(self, model_types=None):
        """Compare multiple trained models."""
        try:
            self.logger.info(f"Starting model comparison. Available models: {list(self.models.keys())}")
            
            if not self.models:
                raise ValueError("No trained models available for comparison")

            # If no specific models requested, compare all trained models
            if model_types is None:
                model_types = list(self.models.keys())
            elif isinstance(model_types, list):
                # Convert model aliases to full names
                model_types = [MODEL_ALIASES.get(mt, mt) for mt in model_types]
            
            self.logger.info(f"Comparing models: {model_types}")

            comparison = {}
            for model_type in model_types:
                if model_type not in self.models:
                    self.logger.warning(f"Model {model_type} not found in trained models")
                    continue
                    
                model_info = self.models[model_type]
                comparison[model_type] = {
                    'metrics': model_info['metrics'],
                    'predictions': model_info['predictions'].tolist() if isinstance(model_info['predictions'], np.ndarray) else model_info['predictions']
                }

            if not comparison:
                raise ValueError("No models found for comparison")

            self.logger.info(f"Comparison completed for {len(comparison)} models")
            return comparison

        except Exception as e:
            self.logger.error(f"Error in model comparison: {str(e)}")
            raise

    def tune_hyperparameters(self, model_type, n_trials=100, cv_folds=5, cv_strategy='kfold'):
        """Perform hyperparameter tuning using Optuna"""
        try:
            if not hasattr(self, 'X') or not hasattr(self, 'y'):
                raise ValueError("Data not loaded. Please load data first.")

            def objective(trial):
                params = self._get_hyperparameter_space(trial, model_type)
                model = self._create_model(model_type, params)

                if cv_strategy == 'kfold':
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                elif cv_strategy == 'stratified':
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                elif cv_strategy == 'timeseries':
                    cv = TimeSeriesSplit(n_splits=cv_folds)
                else:
                    raise ValueError(f"Unknown cross-validation strategy: {cv_strategy}")

                try:
                    scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='neg_mean_squared_error')
                    return -np.mean(scores)  # We minimize the objective
                except Exception as e:
                    print(f"Error during cross-validation: {str(e)}")
                    return float('inf')  # Return worst possible score on error

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)

            return study.best_params

        except Exception as e:
            self.logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise

    def _get_hyperparameter_space(self, trial, model_type):
        """Get hyperparameter space for a given model type"""
        if model_type == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
        elif model_type == 'gradient_boosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
            }
        elif model_type == 'svm':
            return {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True)
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_model(self, model_type, params):
        """Create a model instance with given parameters"""
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            return RandomForestClassifier(**params) if self.is_classification else RandomForestRegressor(**params)
        elif model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            return GradientBoostingClassifier(**params) if self.is_classification else GradientBoostingRegressor(**params)
        elif model_type == 'svm':
            from sklearn.svm import SVC, SVR
            return SVC(**params) if self.is_classification else SVR(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def save_model(self, filepath):
        """Save the trained model"""
        try:
            if self.model is None:
                raise ValueError("No model has been trained yet")
            
            joblib.dump(self.model, filepath)
            return {'message': f'Model saved successfully to {filepath}'}
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        try:
            plots = {}
            
            # Feature Importance Plot
            if self.feature_importance:
                sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
                feature_names = [x[0] for x in sorted_features]
                importance_values = [x[1] for x in sorted_features]
                
                fig = go.Figure(data=[
                    go.Bar(x=importance_values, y=feature_names, orientation='h')
                ])
                fig.update_layout(
                    title='Feature Importance',
                    xaxis_title='Importance Score',
                    yaxis_title='Features',
                    height=max(400, len(feature_names) * 20),
                    margin=dict(l=200, r=40, t=60, b=40)
                )
                plots['feature_importance'] = fig.to_json()

            # Learning Curves
            try:
                train_sizes, train_scores, test_scores = learning_curve(
                    self.model, self.X, self.y,
                    cv=5, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 10)
                )

                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=train_sizes,
                    y=train_mean,
                    mode='lines+markers',
                    name='Training Score',
                    line=dict(color='blue'),
                    error_y=dict(
                        type='data',
                        array=train_std,
                        visible=True
                    )
                ))
                fig.add_trace(go.Scatter(
                    x=train_sizes,
                    y=test_mean,
                    mode='lines+markers',
                    name='Cross-validation Score',
                    line=dict(color='red'),
                    error_y=dict(
                        type='data',
                        array=test_std,
                        visible=True
                    )
                ))
                fig.update_layout(
                    title='Learning Curves',
                    xaxis_title='Training Examples',
                    yaxis_title='Score',
                    height=500
                )
                plots['learning_curves'] = fig.to_json()
            except Exception as e:
                self.logger.warning(f"Could not generate learning curves: {str(e)}")

            # Actual vs Predicted Plot
            try:
                y_pred = self.model.predict(self.X_test)
                
                if self.problem_type == 'regression':
                    fig = go.Figure(data=[
                        go.Scatter(
                            x=self.y_test,
                            y=y_pred,
                            mode='markers',
                            marker=dict(size=8, opacity=0.6),
                            name='Test Data'
                        ),
                        go.Scatter(
                            x=[min(self.y_test), max(self.y_test)],
                            y=[min(self.y_test), max(self.y_test)],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Perfect Prediction'
                        )
                    ])
                    fig.update_layout(
                        title='Actual vs Predicted Values',
                        xaxis_title='Actual Values',
                        yaxis_title='Predicted Values',
                        height=500
                    )
                else:  # Classification
                    if hasattr(self.model, 'predict_proba'):
                        y_proba = self.model.predict_proba(self.X_test)
                        fig = go.Figure()
                        for i in range(y_proba.shape[1]):
                            mask = self.y_test == i
                            fig.add_trace(go.Box(
                                y=y_proba[mask, i],
                                name=f'Class {i}',
                                boxpoints='outliers'
                            ))
                        fig.update_layout(
                            title='Prediction Probabilities by Class',
                            yaxis_title='Prediction Probability',
                            height=500
                        )
                plots['prediction_plot'] = fig.to_json()
            except Exception as e:
                self.logger.warning(f"Could not generate prediction plot: {str(e)}")

            # SHAP Values Plot
            try:
                if self.X_test is not None and len(self.X_test) > 0:
                    explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'estimators_') else shap.KernelExplainer(self.model.predict, shap.sample(self.X_test, 100))
                    shap_values = explainer.shap_values(self.X_test.iloc[:100])  # Limit to 100 samples for performance
                    
                    if isinstance(shap_values, list):  # For multi-class classification
                        shap_values = np.abs(np.array(shap_values)).mean(0)  # Take mean of absolute values across classes
                    
                    feature_importance = np.abs(shap_values).mean(0)
                    feature_importance_dict = dict(zip(self.X_test.columns, feature_importance))
                    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[x[1] for x in sorted_features],
                            y=[x[0] for x in sorted_features],
                            orientation='h'
                        )
                    ])
                    fig.update_layout(
                        title='SHAP Feature Importance',
                        xaxis_title='mean(|SHAP value|)',
                        yaxis_title='Feature',
                        height=max(400, len(sorted_features) * 20),
                        margin=dict(l=200, r=40, t=60, b=40)
                    )
                    plots['shap_importance'] = fig.to_json()
            except Exception as e:
                self.logger.warning(f"Could not generate SHAP plot: {str(e)}")

            return plots
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            raise

    def plt2json(self, plt_figure):
        """Convert matplotlib figure to JSON for web display"""
        try:
            import io
            import base64
            
            buf = io.BytesIO()
            plt_figure.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            string = base64.b64encode(buf.read()).decode('utf-8')
            return f'data:image/png;base64,{string}'
            
        except Exception as e:
            self.logger.error(f"Error converting plot to JSON: {str(e)}")
            raise
