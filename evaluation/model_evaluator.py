"""
Model evaluation system with comprehensive metrics and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    confusion_matrix
)
from datetime import datetime, timedelta

from .config import CONFIG
from .utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float]
    confusion_matrix: np.ndarray
    feature_importance: Optional[Dict[str, float]]
    timestamp: datetime

@dataclass
class RegressionMetrics:
    """Container for regression model metrics."""
    mse: float
    rmse: float
    mae: float
    mape: float
    r2_score: float
    feature_importance: Optional[Dict[str, float]]
    timestamp: datetime

class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self):
        self.metrics_history: List[Union[ModelMetrics, RegressionMetrics]] = []
        self.feature_importance: Optional[Dict[str, float]] = None
        
    def evaluate_classifier(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> ModelMetrics:
        """Evaluate classification model performance."""
        try:
            # Calculate basic metrics
            metrics = ModelMetrics(
                accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, average='weighted'),
                recall=recall_score(y_true, y_pred, average='weighted'),
                f1_score=f1_score(y_true, y_pred, average='weighted'),
                auc_roc=roc_auc_score(y_true, y_prob) if y_prob is not None else None,
                confusion_matrix=confusion_matrix(y_true, y_pred),
                feature_importance=feature_importance,
                timestamp=datetime.now()
            )
            
            # Store metrics in history
            self.metrics_history.append(metrics)
            self.feature_importance = feature_importance
            
            logger.info("Model evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
            
    def evaluate_regressor(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> RegressionMetrics:
        """Evaluate regression model performance."""
        try:
            # Calculate MSE and RMSE
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate MAE
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate MAPE
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Calculate R2 score
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / 
                      np.sum((y_true - np.mean(y_true)) ** 2))
            
            metrics = RegressionMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                mape=mape,
                r2_score=r2,
                feature_importance=feature_importance,
                timestamp=datetime.now()
            )
            
            # Store metrics in history
            self.metrics_history.append(metrics)
            self.feature_importance = feature_importance
            
            logger.info("Regression evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in regression evaluation: {str(e)}")
            raise
            
    def evaluate_feature_importance(
        self,
        model: any,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Evaluate feature importance from model."""
        try:
            # Get feature importance scores
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_)
            else:
                logger.warning("Model does not provide feature importance scores")
                return {}
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, importance_scores))
            
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            self.feature_importance = feature_importance
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise
            
    def calculate_class_metrics(
        self,
        confusion_mat: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics from confusion matrix."""
        try:
            n_classes = confusion_mat.shape[0]
            class_metrics = {}
            
            for i in range(n_classes):
                tp = confusion_mat[i, i]
                fp = np.sum(confusion_mat[:, i]) - tp
                fn = np.sum(confusion_mat[i, :]) - tp
                tn = np.sum(confusion_mat) - (tp + fp + fn)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) \
                    if (precision + recall) > 0 else 0
                
                class_metrics[f'class_{i}'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': np.sum(confusion_mat[i, :])
                }
                
            return class_metrics
            
        except Exception as e:
            logger.error(f"Error calculating class metrics: {str(e)}")
            raise
            
    def generate_evaluation_report(self) -> Dict[str, any]:
        """Generate comprehensive evaluation report."""
        if not self.metrics_history:
            return {"error": "No evaluation metrics available"}
            
        latest_metrics = self.metrics_history[-1]
        
        if isinstance(latest_metrics, ModelMetrics):
            report = {
                'model_type': 'classifier',
                'overall_metrics': {
                    'accuracy': latest_metrics.accuracy,
                    'precision': latest_metrics.precision,
                    'recall': latest_metrics.recall,
                    'f1_score': latest_metrics.f1_score,
                    'auc_roc': latest_metrics.auc_roc
                },
                'class_metrics': self.calculate_class_metrics(
                    latest_metrics.confusion_matrix
                ),
                'feature_importance': self.feature_importance,
                'metrics_history': self._analyze_metrics_history()
            }
        else:
            report = {
                'model_type': 'regressor',
                'overall_metrics': {
                    'mse': latest_metrics.mse,
                    'rmse': latest_metrics.rmse,
                    'mae': latest_metrics.mae,
                    'mape': latest_metrics.mape,
                    'r2_score': latest_metrics.r2_score
                },
                'feature_importance': self.feature_importance,
                'metrics_history': self._analyze_metrics_history()
            }
            
        return report
        
    def _analyze_metrics_history(self) -> Dict[str, any]:
        """Analyze trends in evaluation metrics."""
        if len(self.metrics_history) < 2:
            return {}
            
        trends = {}
        
        if isinstance(self.metrics_history[0], ModelMetrics):
            metrics_to_track = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in metrics_to_track:
                values = [getattr(m, metric) for m in self.metrics_history]
                trends[metric] = {
                    'trend': np.polyfit(range(len(values)), values, 1)[0],
                    'volatility': np.std(values),
                    'current_vs_average': values[-1] / np.mean(values)
                }
        else:
            metrics_to_track = ['mse', 'rmse', 'mae', 'mape']
            for metric in metrics_to_track:
                values = [getattr(m, metric) for m in self.metrics_history]
                trends[metric] = {
                    'trend': np.polyfit(range(len(values)), values, 1)[0],
                    'volatility': np.std(values),
                    'current_vs_average': values[-1] / np.mean(values)
                }
                
        return trends