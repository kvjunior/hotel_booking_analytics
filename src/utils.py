"""
Utility functions for the hotel booking analytics system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Union
import json
import yaml
from datetime import datetime, date
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from .config import CONFIG

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling special data types."""
    
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)

def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """Set up and configure logger."""
    logger = logging.getLogger(name or __name__)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(CONFIG['LOG_FORMAT'])
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Also add a file handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f'{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.setLevel(CONFIG['LOG_LEVEL'])
    
    return logger

logger = setup_logger(__name__)

def load_yaml_config(filepath: Union[str, Path]) -> Dict:
    """Load YAML configuration file."""
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {filepath}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def save_json_results(
    data: Dict,
    filepath: Union[str, Path],
    indent: int = 2
) -> None:
    """Save results to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, cls=CustomJSONEncoder, indent=indent)
        logger.info(f"Successfully saved results to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def create_data_splits(
    df: pd.DataFrame,
    target_col: str,
    stratify: bool = True
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """Create train, validation, and test splits."""
    try:
        # First split: separate test set
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        stratify_col = y if stratify else None
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=CONFIG['TEST_SIZE'],
            random_state=CONFIG['RANDOM_STATE'],
            stratify=stratify_col
        )
        
        # Second split: separate validation set from training set
        stratify_col = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=CONFIG['VALIDATION_SIZE'],
            random_state=CONFIG['RANDOM_STATE'],
            stratify=stratify_col
        )
        
        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        logger.info("Successfully created data splits")
        return splits
        
    except Exception as e:
        logger.error(f"Error creating data splits: {str(e)}")
        raise

def calculate_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_prob: Optional[Union[List, np.ndarray]] = None
) -> Dict[str, float]:
    """Calculate various performance metrics."""
    try:
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Calculate AUC if probabilities are provided
        if y_prob is not None:
            y_prob = np.array(y_prob)
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Union[str, Path]
) -> None:
    """Plot training history for neural networks."""
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Training history plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")
        raise

def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    output_path: Union[str, Path],
    top_n: int = 20
) -> None:
    """Plot feature importance scores."""
    try:
        # Sort features by importance
        indices = np.argsort(importance_scores)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importance_scores[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise

def create_correlation_matrix(
    df: pd.DataFrame,
    output_path: Union[str, Path]
) -> None:
    """Create and plot correlation matrix."""
    try:
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Correlation matrix plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {str(e)}")
        raise

def handle_missing_values(
    df: pd.DataFrame,
    strategy: Dict[str, str]
) -> pd.DataFrame:
    """Handle missing values in DataFrame."""
    try:
        df_clean = df.copy()
        
        for column, method in strategy.items():
            if column in df.columns:
                if method == 'mean':
                    df_clean[column].fillna(df[column].mean(), inplace=True)
                elif method == 'median':
                    df_clean[column].fillna(df[column].median(), inplace=True)
                elif method == 'mode':
                    df_clean[column].fillna(df[column].mode()[0], inplace=True)
                elif method == 'drop':
                    df_clean.dropna(subset=[column], inplace=True)
                    
        logger.info("Successfully handled missing values")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
        raise

def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> bool:
    """Validate data types of DataFrame columns."""
    try:
        for column, expected_type in expected_types.items():
            if column in df.columns:
                actual_type = df[column].dtype
                if str(actual_type) != expected_type:
                    logger.warning(
                        f"Data type mismatch in column {column}: "
                        f"expected {expected_type}, got {actual_type}"
                    )
                    return False
        return True
        
    except Exception as e:
        logger.error(f"Error validating data types: {str(e)}")
        raise

def create_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Create time-based features from date column."""
    try:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Extract time components
        df[f"{date_column}_year"] = df[date_column].dt.year
        df[f"{date_column}_month"] = df[date_column].dt.month
        df[f"{date_column}_day"] = df[date_column].dt.day
        df[f"{date_column}_dayofweek"] = df[date_column].dt.dayofweek
        df[f"{date_column}_quarter"] = df[date_column].dt.quarter
        df[f"{date_column}_is_weekend"] = df[date_column].dt.dayofweek.isin([5, 6])
        
        logger.info(f"Successfully created time features from {date_column}")
        return df
        
    except Exception as e:
        logger.error(f"Error creating time features: {str(e)}")
        raise