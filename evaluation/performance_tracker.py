"""
Performance tracking system for monitoring and analyzing model performance over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

from .config import CONFIG
from .utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance tracking metrics."""
    model_id: str
    accuracy: float
    latency: float
    memory_usage: float
    prediction_variance: float
    data_drift_score: float
    timestamp: datetime

@dataclass
class ModelHealth:
    """Container for model health metrics."""
    status: str
    health_score: float
    issues: List[str]
    recommendations: List[str]

class PerformanceTracker:
    """Tracks and analyzes model performance over time."""
    
    def __init__(self):
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.health_checks: Dict[str, ModelHealth] = {}
        self.alerts_history: List[Dict] = []
        
    def track_performance(
        self,
        model_id: str,
        predictions: np.ndarray,
        actual: np.ndarray,
        latency: float,
        memory_usage: float
    ) -> PerformanceMetrics:
        """Track model performance metrics."""
        try:
            # Calculate accuracy/error
            if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                # Regression case
                accuracy = 1 - mean_squared_error(actual, predictions)
            else:
                # Classification case
                accuracy = np.mean(predictions.argmax(axis=1) == actual)
                
            # Calculate prediction variance
            prediction_variance = np.var(predictions)
            
            # Calculate data drift score
            data_drift_score = self._calculate_data_drift(predictions)
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                model_id=model_id,
                accuracy=accuracy,
                latency=latency,
                memory_usage=memory_usage,
                prediction_variance=prediction_variance,
                data_drift_score=data_drift_score,
                timestamp=datetime.now()
            )
            
            # Store metrics in history
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            self.performance_history[model_id].append(metrics)
            
            # Check for performance issues
            self._check_performance_issues(model_id, metrics)
            
            logger.info(f"Performance tracked for model {model_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error tracking performance: {str(e)}")
            raise
            
    def check_model_health(
        self,
        model_id: str,
        threshold_config: Optional[Dict] = None
    ) -> ModelHealth:
        """Check overall model health."""
        try:
            if model_id not in self.performance_history:
                raise ValueError(f"No performance history for model {model_id}")
                
            metrics = self.performance_history[model_id][-1]
            thresholds = threshold_config or {
                'min_accuracy': 0.7,
                'max_latency': 1.0,
                'max_memory': 1000,
                'max_drift': 0.3
            }
            
            # Initialize health check
            issues = []
            recommendations = []
            
            # Check accuracy
            if metrics.accuracy < thresholds['min_accuracy']:
                issues.append("Low model accuracy")
                recommendations.append("Consider retraining the model")
                
            # Check latency
            if metrics.latency > thresholds['max_latency']:
                issues.append("High prediction latency")
                recommendations.append("Optimize model for faster inference")
                
            # Check memory usage
            if metrics.memory_usage > thresholds['max_memory']:
                issues.append("High memory usage")
                recommendations.append("Consider model compression techniques")
                
            # Check data drift
            if metrics.data_drift_score > thresholds['max_drift']:
                issues.append("Significant data drift detected")
                recommendations.append("Update model with recent data")
                
            # Calculate health score
            health_score = self._calculate_health_score(metrics, thresholds)
            
            # Determine status
            if health_score > 0.8:
                status = "Healthy"
            elif health_score > 0.6:
                status = "Warning"
            else:
                status = "Critical"
                
            # Create health check result
            health = ModelHealth(
                status=status,
                health_score=health_score,
                issues=issues,
                recommendations=recommendations
            )
            
            # Store health check
            self.health_checks[model_id] = health
            
            logger.info(f"Health check completed for model {model_id}")
            return health
            
        except Exception as e:
            logger.error(f"Error checking model health: {str(e)}")
            raise
            
    def _calculate_data_drift(self,
        predictions: np.ndarray
    ) -> float:
        """Calculate data drift score based on prediction distribution."""
        try:
            # Calculate distribution metrics
            current_mean = np.mean(predictions)
            current_std = np.std(predictions)
            
            # If we have historical predictions, compare distributions
            if hasattr(self, 'historical_stats'):
                mean_diff = abs(current_mean - self.historical_stats['mean'])
                std_diff = abs(current_std - self.historical_stats['std'])
                
                # Calculate drift score (0 to 1)
                drift_score = (mean_diff / self.historical_stats['mean'] + 
                             std_diff / self.historical_stats['std']) / 2
            else:
                # Initialize historical stats
                self.historical_stats = {
                    'mean': current_mean,
                    'std': current_std
                }
                drift_score = 0.0
                
            return drift_score
            
        except Exception as e:
            logger.error(f"Error calculating data drift: {str(e)}")
            return 0.0
            
    def _calculate_health_score(
        self,
        metrics: PerformanceMetrics,
        thresholds: Dict[str, float]
    ) -> float:
        """Calculate overall model health score."""
        try:
            # Component scores
            accuracy_score = metrics.accuracy / thresholds['min_accuracy']
            latency_score = 1 - (metrics.latency / thresholds['max_latency'])
            memory_score = 1 - (metrics.memory_usage / thresholds['max_memory'])
            drift_score = 1 - (metrics.data_drift_score / thresholds['max_drift'])
            
            # Weights for different components
            weights = {
                'accuracy': 0.4,
                'latency': 0.2,
                'memory': 0.2,
                'drift': 0.2
            }
            
            # Calculate weighted score
            health_score = (
                weights['accuracy'] * accuracy_score +
                weights['latency'] * latency_score +
                weights['memory'] * memory_score +
                weights['drift'] * drift_score
            )
            
            return min(max(health_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating health score: {str(e)}")
            return 0.0
            
    def _check_performance_issues(
        self,
        model_id: str,
        metrics: PerformanceMetrics
    ):
        """Check for performance issues and create alerts."""
        try:
            # Define thresholds
            thresholds = {
                'accuracy_drop': 0.1,
                'latency_increase': 0.5,
                'memory_increase': 0.3,
                'drift_threshold': 0.2
            }
            
            # Get historical metrics
            if len(self.performance_history[model_id]) > 1:
                prev_metrics = self.performance_history[model_id][-2]
                
                # Check for significant drops in accuracy
                if (prev_metrics.accuracy - metrics.accuracy) > thresholds['accuracy_drop']:
                    self._create_alert(
                        model_id,
                        "Accuracy Drop",
                        f"Accuracy dropped by {(prev_metrics.accuracy - metrics.accuracy):.2%}"
                    )
                    
                # Check for latency increases
                if (metrics.latency / prev_metrics.latency - 1) > thresholds['latency_increase']:
                    self._create_alert(
                        model_id,
                        "High Latency",
                        f"Latency increased by {(metrics.latency / prev_metrics.latency - 1):.2%}"
                    )
                    
                # Check for memory usage increases
                if (metrics.memory_usage / prev_metrics.memory_usage - 1) > thresholds['memory_increase']:
                    self._create_alert(
                        model_id,
                        "High Memory Usage",
                        f"Memory usage increased by {(metrics.memory_usage / prev_metrics.memory_usage - 1):.2%}"
                    )
                    
            # Check for data drift
            if metrics.data_drift_score > thresholds['drift_threshold']:
                self._create_alert(
                    model_id,
                    "Data Drift",
                    f"Data drift score: {metrics.data_drift_score:.2f}"
                )
                
        except Exception as e:
            logger.error(f"Error checking performance issues: {str(e)}")
            
    def _create_alert(
        self,
        model_id: str,
        alert_type: str,
        message: str
    ):
        """Create and store performance alert."""
        alert = {
            'model_id': model_id,
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now()
        }
        
        self.alerts_history.append(alert)
        logger.warning(f"Alert for model {model_id}: {message}")
        
    def generate_performance_report(
        self,
        model_id: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, any]:
        """Generate comprehensive performance report."""
        try:
            if model_id not in self.performance_history:
                return {"error": f"No performance history for model {model_id}"}
                
            # Filter metrics by time window
            metrics = self.performance_history[model_id]
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
                
            # Calculate statistics
            accuracy_values = [m.accuracy for m in metrics]
            latency_values = [m.latency for m in metrics]
            memory_values = [m.memory_usage for m in metrics]
            drift_values = [m.data_drift_score for m in metrics]
            
            report = {
                'model_id': model_id,
                'time_period': {
                    'start': metrics[0].timestamp,
                    'end': metrics[-1].timestamp
                },
                'performance_metrics': {
                    'accuracy': {
                        'current': accuracy_values[-1],
                        'mean': np.mean(accuracy_values),
                        'std': np.std(accuracy_values),
                        'trend': np.polyfit(range(len(accuracy_values)), accuracy_values, 1)[0]
                    },
                    'latency': {
                        'current': latency_values[-1],
                        'mean': np.mean(latency_values),
                        'std': np.std(latency_values),
                        'trend': np.polyfit(range(len(latency_values)), latency_values, 1)[0]
                    },
                    'memory_usage': {
                        'current': memory_values[-1],
                        'mean': np.mean(memory_values),
                        'std': np.std(memory_values),
                        'trend': np.polyfit(range(len(memory_values)), memory_values, 1)[0]
                    },
                    'data_drift': {
                        'current': drift_values[-1],
                        'mean': np.mean(drift_values),
                        'std': np.std(drift_values),
                        'trend': np.polyfit(range(len(drift_values)), drift_values, 1)[0]
                    }
                },
                'health_check': self.health_checks.get(model_id, None),
                'alerts': [
                    alert for alert in self.alerts_history
                    if alert['model_id'] == model_id
                ],
                'recommendations': self._generate_recommendations(model_id)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            raise
            
    def _generate_recommendations(self, model_id: str) -> List[str]:
        """Generate recommendations based on performance history."""
        recommendations = []
        
        try:
            metrics = self.performance_history[model_id]
            recent_metrics = metrics[-10:]  # Look at last 10 measurements
            
            # Check accuracy trend
            accuracy_trend = np.polyfit(
                range(len(recent_metrics)),
                [m.accuracy for m in recent_metrics],
                1
            )[0]
            
            if accuracy_trend < -0.01:
                recommendations.append(
                    "Model accuracy is declining. Consider retraining with recent data."
                )
                
            # Check latency
            avg_latency = np.mean([m.latency for m in recent_metrics])
            if avg_latency > 0.5:
                recommendations.append(
                    "High average latency. Consider model optimization or hardware upgrades."
                )
                
            # Check memory usage trend
            memory_trend = np.polyfit(
                range(len(recent_metrics)),
                [m.memory_usage for m in recent_metrics],
                1
            )[0]
            
            if memory_trend > 0.1:
                recommendations.append(
                    "Memory usage is increasing. Consider model compression or cleanup."
                )
                
            # Check data drift
            avg_drift = np.mean([m.data_drift_score for m in recent_metrics])
            if avg_drift > 0.2:
                recommendations.append(
                    "Significant data drift detected. Validate input data distribution."
                )
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations due to an error.