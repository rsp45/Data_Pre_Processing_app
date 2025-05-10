import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, abort
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import tempfile
import json
import logging
import traceback
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine, fetch_california_housing
from ml_processor import MLProcessor, MODEL_ALIASES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.urandom(24)

# Global ML processor instance
ml_processor = None

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global ml_processor
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            })
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            })

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Initialize MLProcessor with the uploaded file
        try:
            ml_processor = MLProcessor(data_path=filepath)
            
            return jsonify({
                'status': 'success',
                'message': 'File uploaded successfully',
                'columns': ml_processor.data.columns.tolist(),
                'shape': ml_processor.data.shape
            })
            
        except Exception as e:
            # Clean up the uploaded file if processing fails
            if os.path.exists(filepath):
                os.remove(filepath)
            raise ValueError(f"Error processing file: {str(e)}")

    except Exception as e:
        app.logger.error(f"Error uploading file: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error uploading file: {str(e)}"
        })

@app.route('/load_example_dataset/<dataset_name>', methods=['GET'])
def load_example_dataset(dataset_name):
    try:
        # Load the selected dataset
        if dataset_name == 'iris':
            data = load_iris()
        elif dataset_name == 'california':
            data = fetch_california_housing()
        elif dataset_name == 'diabetes':
            data = load_diabetes()
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
        elif dataset_name == 'wine':
            data = load_wine()
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid dataset name'
            })

        # Convert to pandas DataFrame
        if hasattr(data, 'data'):
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        else:
            df = pd.DataFrame(data)

        # Initialize MLProcessor
        global ml_processor
        ml_processor = MLProcessor(data=df)

        return jsonify({
            'status': 'success',
            'message': f'Loaded {dataset_name} dataset',
            'shape': df.shape,
            'columns': df.columns.tolist()
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/detect_problem_type', methods=['POST'])
def detect_problem():
    try:
        data = request.get_json()
        target_column = data.get('target_column')
        
        if not ml_processor:
            return jsonify({
                'status': 'error',
                'message': 'Please upload or select a dataset first'
            })
        
        problem_type = ml_processor.detect_problem_type(target_column)
        
        return jsonify({
            'status': 'success',
            'problem_type': problem_type
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/analyze_bias', methods=['POST'])
def analyze_bias():
    try:
        if not ml_processor:
            return jsonify({
                'status': 'error',
                'message': 'Please upload or select a dataset first'
            })
        
        data = request.get_json()
        if not data or 'target_column' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Target column not specified'
            })
        
        target_column = data['target_column']
        
        # First detect problem type
        problem_type = ml_processor.detect_problem_type(target_column)
        
        # Then analyze bias
        bias_report = ml_processor.analyze_data_bias()
        
        return jsonify({
            'status': 'success',
            'problem_type': problem_type,
            'bias_report': bias_report
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        if not ml_processor:
            return jsonify({
                'status': 'error',
                'message': 'Please upload or select a dataset first'
            })
        
        data = request.get_json()
        if not data or 'target_column' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Target column not specified'
            })
        
        test_size = data.get('test_size', 0.2)
        handle_imbalance = data.get('handle_imbalance', True)
        
        # Preprocess the data
        preprocessing_results = ml_processor.preprocess_data(
            test_size=test_size,
            handle_imbalance=handle_imbalance
        )
        
        # Ensure we have the expected structure
        if not preprocessing_results:
            preprocessing_results = {}
        
        # Add basic shape information if not present
        if 'train_shape' not in preprocessing_results and hasattr(ml_processor, 'X_train'):
            preprocessing_results['train_shape'] = {
                'X': list(ml_processor.X_train.shape),
                'y': list(np.shape(ml_processor.y_train))
            }
        if 'test_shape' not in preprocessing_results and hasattr(ml_processor, 'X_test'):
            preprocessing_results['test_shape'] = {
                'X': list(ml_processor.X_test.shape),
                'y': list(np.shape(ml_processor.y_test))
            }
        
        return jsonify({
            'status': 'success',
            'preprocessing_results': preprocessing_results
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        app.logger.info("Received training request")
        
        if not ml_processor:
            return jsonify({
                'status': 'error',
                'message': 'Please upload or select a dataset first'
            })

        data = request.get_json()
        model_type = data.get('model_type')
        
        app.logger.info(f"Requested model type: {model_type}")
        
        if not model_type:
            return jsonify({
                'status': 'error',
                'message': 'Model type not specified'
            })

        # Map model alias to actual model type
        if model_type in MODEL_ALIASES:
            original_type = model_type
            model_type = MODEL_ALIASES[model_type]
            app.logger.info(f"Mapped model type from {original_type} to {model_type}")

        # Train the model and get results
        app.logger.info("Starting model training...")
        results = ml_processor.train_model(model_type)
        
        # Log the complete results
        app.logger.info("Training completed. Results:")
        app.logger.info(f"Status: {results.get('status')}")
        app.logger.info(f"Message: {results.get('message')}")
        app.logger.info(f"Train shape: {results.get('train_shape')}")
        app.logger.info(f"Test shape: {results.get('test_shape')}")
        
        return jsonify(results)

    except Exception as e:
        app.logger.error(f"Error in train_model endpoint: {str(e)}")
        app.logger.error(f"Stack trace: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/tune_hyperparameters', methods=['POST'])
def tune_hyperparameters():
    try:
        data = request.get_json()
        n_trials = data.get('n_trials', 100)
        cv_folds = data.get('cv_folds', 5)
        cv_strategy = data.get('cv_strategy', 'kfold')
        model_type = data.get('model_type')

        if not model_type:
            return jsonify({'error': 'Model type is required'}), 400

        best_params = ml_processor.tune_hyperparameters(
            model_type=model_type,
            n_trials=n_trials,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy
        )

        return jsonify({'best_params': best_params})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualizations', methods=['GET'])
def get_visualizations():
    try:
        if not ml_processor:
            return jsonify({
                'status': 'error',
                'message': 'Please upload or select a dataset first'
            })
        
        plots = ml_processor.create_visualizations()
        
        return jsonify({
            'status': 'success',
            'plots': plots
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/save_model', methods=['POST'])
def save():
    try:
        data = request.get_json()
        model_name = data.get('model_name', 'model.joblib')
        
        if not ml_processor:
            return jsonify({
                'status': 'error',
                'message': 'Please upload or select a dataset first'
            })
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(models_dir, model_name)
        save_result = ml_processor.save_model(model_path)
        
        return jsonify({
            'status': 'success',
            'message': save_result['message']
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/advanced_eda', methods=['GET'])
def advanced_eda():
    try:
        if ml_processor is None or ml_processor.data is None:
            return jsonify({
                'status': 'error',
                'message': 'No data loaded. Please upload a dataset first.'
            })

        data = ml_processor.data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns

        # Basic statistics for numeric columns
        numeric_stats = data[numeric_cols].describe().round(2).to_dict()

        # Correlation matrix
        correlation_matrix = data[numeric_cols].corr().round(3)
        correlation_plot = {
            'data': [{
                'type': 'heatmap',
                'z': correlation_matrix.values.tolist(),
                'x': correlation_matrix.columns.tolist(),
                'y': correlation_matrix.columns.tolist(),
                'colorscale': 'RdBu',
                'zmin': -1,
                'zmax': 1,
                'colorbar': {'title': 'Correlation'}
            }],
            'layout': {
                'title': 'Correlation Matrix',
                'width': 800,
                'height': 800,
                'xaxis': {'tickangle': 45},
                'margin': {'l': 150, 'r': 50, 't': 50, 'b': 150}
            }
        }

        # Distribution plots for numeric columns
        distribution_plots = {}
        for col in numeric_cols:
            data_col = data[col].dropna()
            if len(data_col) > 0:
                # Create histogram with KDE
                hist_values, bin_edges = np.histogram(data_col, bins='auto', density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Calculate KDE
                kde = gaussian_kde(data_col)
                x_range = np.linspace(min(data_col), max(data_col), 100)
                kde_values = kde(x_range)

                distribution_plots[col] = {
                    'data': [
                        {
                            'type': 'histogram',
                            'x': data_col.tolist(),
                            'name': 'Histogram',
                            'histnorm': 'probability density',
                            'opacity': 0.7
                        },
                        {
                            'type': 'scatter',
                            'x': x_range.tolist(),
                            'y': kde_values.tolist(),
                            'name': 'KDE',
                            'line': {'color': 'red'}
                        }
                    ],
                    'layout': {
                        'title': f'Distribution of {col}',
                        'xaxis': {'title': col},
                        'yaxis': {'title': 'Density'},
                        'showlegend': True,
                        'bargap': 0.1
                    }
                }

        # Box plots for numeric columns
        box_plots = {
            'data': [
                {
                    'type': 'box',
                    'y': data[col].dropna().tolist(),
                    'name': col,
                    'boxpoints': 'outliers'
                } for col in numeric_cols
            ],
            'layout': {
                'title': 'Box Plots of Numeric Features',
                'yaxis': {'title': 'Value'},
                'showlegend': False,
                'height': max(400, len(numeric_cols) * 50)
            }
        }

        # Bar plots for categorical columns
        categorical_plots = {}
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            categorical_plots[col] = {
                'data': [{
                    'type': 'bar',
                    'x': value_counts.index.tolist(),
                    'y': value_counts.values.tolist(),
                    'marker': {'color': 'rgb(79, 70, 229)'}
                }],
                'layout': {
                    'title': f'Distribution of {col}',
                    'xaxis': {'title': col, 'tickangle': 45},
                    'yaxis': {'title': 'Count'},
                    'margin': {'b': 100}
                }
            }

        return jsonify({
            'status': 'success',
            'numeric_stats': numeric_stats,
            'correlation_plot': correlation_plot,
            'distribution_plots': distribution_plots,
            'box_plots': box_plots,
            'categorical_plots': categorical_plots
        })

    except Exception as e:
        app.logger.error(f"Error in advanced_eda: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/post_eda', methods=['GET'])
def post_eda():
    """Perform EDA on preprocessed data"""
    try:
        if not ml_processor:
            return jsonify({
                'status': 'error',
                'message': 'Please upload or select a dataset first'
            })
        
        if not hasattr(ml_processor, 'X_train') or not hasattr(ml_processor, 'y_train'):
            return jsonify({
                'status': 'error',
                'message': 'Please preprocess the data first'
            })
            
        # Get feature correlations
        correlations = {}
        if ml_processor.X_train.select_dtypes(include=['int64', 'float64']).columns.size > 0:
            corr_matrix = ml_processor.X_train.corr()
            correlations = {
                'matrix': corr_matrix.to_dict(),
                'features': corr_matrix.columns.tolist()
            }
        
        # Get feature importance if classification
        feature_importance = None
        if ml_processor.problem_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(ml_processor.X_train, ml_processor.y_train)
            importance = rf.feature_importances_
            feature_importance = {
                str(col): float(imp) for col, imp in 
                zip(ml_processor.X_train.columns, importance)
            }
        
        # Get distribution plots for numerical features
        distributions = {}
        numerical_features = ml_processor.X_train.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_features:
            train_data = ml_processor.X_train[col].tolist()
            test_data = ml_processor.X_test[col].tolist()
            distributions[str(col)] = {
                'train': {
                    'mean': float(np.mean(train_data)),
                    'std': float(np.std(train_data)),
                    'min': float(np.min(train_data)),
                    'max': float(np.max(train_data)),
                    'data': train_data[:1000]  # Limit data points for visualization
                },
                'test': {
                    'mean': float(np.mean(test_data)),
                    'std': float(np.std(test_data)),
                    'min': float(np.min(test_data)),
                    'max': float(np.max(test_data)),
                    'data': test_data[:1000]  # Limit data points for visualization
                }
            }
        
        # Get target distribution
        target_distribution = None
        if ml_processor.problem_type == 'classification':
            train_dist = pd.Series(ml_processor.y_train).value_counts(normalize=True)
            test_dist = pd.Series(ml_processor.y_test).value_counts(normalize=True)
            target_distribution = {
                'train': {str(k): float(v) for k, v in train_dist.items()},
                'test': {str(k): float(v) for k, v in test_dist.items()}
            }
        else:
            target_distribution = {
                'train': {
                    'mean': float(np.mean(ml_processor.y_train)),
                    'std': float(np.std(ml_processor.y_train)),
                    'min': float(np.min(ml_processor.y_train)),
                    'max': float(np.max(ml_processor.y_train)),
                    'data': ml_processor.y_train[:1000].tolist()
                },
                'test': {
                    'mean': float(np.mean(ml_processor.y_test)),
                    'std': float(np.std(ml_processor.y_test)),
                    'min': float(np.min(ml_processor.y_test)),
                    'max': float(np.max(ml_processor.y_test)),
                    'data': ml_processor.y_test[:1000].tolist()
                }
            }
        
        return jsonify({
            'status': 'success',
            'eda_results': {
                'correlations': correlations,
                'feature_importance': feature_importance,
                'distributions': distributions,
                'target_distribution': target_distribution,
                'problem_type': ml_processor.problem_type
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/perform_eda', methods=['POST'])
def perform_eda():
    try:
        if not ml_processor:
            return jsonify({
                'status': 'error',
                'message': 'No data loaded. Please upload data first.'
            })
            
        # Perform EDA
        eda_results = ml_processor.perform_eda()
        
        # Convert numpy values to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                              np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Process the results to ensure they're JSON serializable
        processed_results = {
            k: convert_to_serializable(v) if isinstance(v, (np.generic, np.ndarray)) else v 
            for k, v in eda_results.items()
        }
        
        return jsonify(processed_results)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/preprocess_data', methods=['POST'])
def preprocess_data():
    try:
        if not ml_processor:
            return jsonify({
                'status': 'error',
                'message': 'No data loaded. Please upload data first.'
            })
            
        # Preprocess the data
        preprocessing_results = ml_processor.preprocess_data()
        
        # Convert numpy values to native Python types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                              np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Process the results to ensure they're JSON serializable
        processed_results = {
            k: convert_to_serializable(v) if isinstance(v, (np.generic, np.ndarray)) else v 
            for k, v in preprocessing_results.items()
        }
        
        return jsonify(processed_results)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/set_target', methods=['POST'])
def set_target():
    try:
        if not ml_processor:
            return jsonify({
                'status': 'error',
                'message': 'Please upload or select a dataset first'
            })

        data = request.get_json()
        target_column = data.get('target_column')
        
        if not target_column:
            return jsonify({
                'status': 'error',
                'message': 'Target column not specified'
            })

        # Set target column in MLProcessor
        ml_processor.set_target(target_column)

        # Get problem type and basic target analysis
        target_data = ml_processor.data[target_column]
        
        target_analysis = {
            'type': ml_processor.problem_type,
            'unique_values': int(target_data.nunique()),
            'missing_values': int(target_data.isnull().sum())
        }

        # For classification, add class distribution
        if ml_processor.problem_type == 'classification':
            class_dist = target_data.value_counts().to_dict()
            target_analysis['class_distribution'] = {str(k): int(v) for k, v in class_dist.items()}

        return jsonify({
            'status': 'success',
            'problem_type': ml_processor.problem_type,
            'target_analysis': target_analysis
        })

    except Exception as e:
        app.logger.error(f"Error in set_target: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/compare_models', methods=['POST'])
def compare_models():
    try:
        if not ml_processor:
            return jsonify({
                'status': 'error',
                'message': 'Please upload or select a dataset first'
            })

        data = request.get_json()
        model_types = data.get('model_types', None)  # If None, compare all trained models
        
        app.logger.info(f"Comparing models: {model_types}")
        
        try:
            # Get model comparison
            comparison = ml_processor.get_model_comparison(model_types)
            
            # Create comparison plots
            plots = create_comparison_plots(comparison)
            
            return jsonify({
                'status': 'success',
                'comparison': comparison,
                'plots': plots
            })
        except ValueError as ve:
            return jsonify({
                'status': 'error',
                'message': str(ve)
            })
        
    except Exception as e:
        app.logger.error(f"Error in compare_models: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

def create_comparison_plots(comparison):
    """Create comparison plots for multiple models."""
    try:
        plots = []
        
        # Extract metrics for all models
        metrics_data = {
            model_name: model_info['metrics'] 
            for model_name, model_info in comparison.items()
        }
        
        # Create bar plot for each metric
        common_metrics = set.intersection(*[set(model['metrics'].keys()) for model in comparison.values()])
        
        for metric in common_metrics:
            metric_values = {
                model_name: metrics[metric]
                for model_name, metrics in metrics_data.items()
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(metric_values.keys()),
                    y=list(metric_values.values()),
                    text=[f"{v:.4f}" for v in metric_values.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title=f"Model Comparison - {metric.upper()}",
                xaxis_title="Model",
                yaxis_title=metric.upper(),
                template="plotly_white"
            )
            
            plots.append({
                'type': 'bar',
                'metric': metric,
                'data': fig.to_json()
            })
            
        return plots
        
    except Exception as e:
        app.logger.error(f"Error creating comparison plots: {str(e)}")
        return []

@app.route('/available_datasets')
def available_datasets():
    try:
        datasets = {
            'iris': {
                'name': 'Iris Dataset',
                'description': 'Classic flower classification dataset',
                'type': 'Classification',
                'features': 4,
                'samples': 150
            },
            'california': {
                'name': 'California Housing',
                'description': 'House price prediction dataset',
                'type': 'Regression',
                'features': 8,
                'samples': 20640
            },
            'diabetes': {
                'name': 'Diabetes Dataset',
                'description': 'Disease progression prediction',
                'type': 'Regression',
                'features': 10,
                'samples': 442
            },
            'breast_cancer': {
                'name': 'Breast Cancer Dataset',
                'description': 'Cancer diagnosis classification',
                'type': 'Classification',
                'features': 30,
                'samples': 569
            },
            'wine': {
                'name': 'Wine Dataset',
                'description': 'Wine variety classification',
                'type': 'Classification',
                'features': 13,
                'samples': 178
            }
        }
        return jsonify({
            'status': 'success',
            'datasets': datasets
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
