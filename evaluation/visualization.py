"""
Visualization utilities for model performance and analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

from .config import CONFIG
from .utils import setup_logger

logger = setup_logger(__name__)

class VisualizationManager:
    """Manager for creating and managing visualizations."""
    
    def __init__(self):
        self.style_config = {
            'colormap': 'viridis',
            'template': 'plotly_white',
            'width': 1000,
            'height': 600
        }
        
    def plot_performance_metrics(
        self,
        metrics_history: List[Dict],
        output_path: Optional[str] = None
    ) -> go.Figure:
        """Plot model performance metrics over time."""
        try:
            # Extract timestamps and metrics
            timestamps = [m['timestamp'] for m in metrics_history]
            accuracy = [m['accuracy'] for m in metrics_history]
            latency = [m['latency'] for m in metrics_history]
            memory = [m['memory_usage'] for m in metrics_history]
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Accuracy', 'Latency', 'Memory Usage'),
                shared_xaxes=True
            )
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=timestamps, y=accuracy, name='Accuracy',
                          line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=latency, name='Latency',
                          line=dict(color='red')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=memory, name='Memory Usage',
                          line=dict(color='green')),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=900,
                width=self.style_config['width'],
                template=self.style_config['template'],
                showlegend=True,
                title_text="Model Performance Metrics Over Time"
            )
            
            if output_path:
                fig.write_html(output_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting performance metrics: {str(e)}")
            raise
            
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> go.Figure:
        """Plot confusion matrix."""
        try:
            if labels is None:
                labels = [f'Class {i}' for i in range(len(confusion_matrix))]
                
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=confusion_matrix,
                x=labels,
                y=labels,
                colorscale='RdBu',
                text=confusion_matrix,
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            # Update layout
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                width=800,
                height=800,
                template=self.style_config['template']
            )
            
            if output_path:
                fig.write_html(output_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
            
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        output_path: Optional[str] = None
    ) -> go.Figure:
        """Plot feature importance scores."""
        try:
            # Sort features by importance
            sorted_features = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            # Create bar plot
            fig = go.Figure(data=go.Bar(
                x=list(sorted_features.keys()),
                y=list(sorted_features.values()),
                marker_color='rgb(55, 83, 109)'
            ))
            
            # Update layout
            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Features',
                yaxis_title='Importance Score',
                width=self.style_config['width'],
                height=self.style_config['height'],
                template=self.style_config['template'],
                xaxis_tickangle=-45
            )
            
            if output_path:
                fig.write_html(output_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise
            
    def plot_prediction_distribution(
        self,
        predictions: np.ndarray,
        actual: Optional[np.ndarray] = None,
        output_path: Optional[str] = None
    ) -> go.Figure:
        """Plot distribution of predictions vs actual values."""
        try:
            fig = go.Figure()
            
            # Add prediction distribution
            fig.add_trace(go.Histogram(
                x=predictions,
                name='Predictions',
                opacity=0.7,
                nbinsx=30
            ))
            
            # Add actual distribution if available
            if actual is not None:
                fig.add_trace(go.Histogram(
                    x=actual,
                    name='Actual',
                    opacity=0.7,
                    nbinsx=30
                ))
                
            # Update layout
            fig.update_layout(
                title='Prediction Distribution',
                xaxis_title='Value',
                yaxis_title='Count',
                barmode='overlay',
                width=self.style_config['width'],
                height=self.style_config['height'],
                template=self.style_config['template']
            )
            
            if output_path:
                fig.write_html(output_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting prediction distribution: {str(e)}")
            raise
            
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        output_path: Optional[str] = None
    ) -> go.Figure:
        """Plot ROC curve with AUC score."""
        try:
            fig = go.Figure()
            
            # Add ROC curve
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC curve (AUC = {auc_score:.3f})',
                line=dict(color='blue')
            ))
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='red', dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=800,
                height=800,
                template=self.style_config['template']
            )
            
            if output_path:
                fig.write_html(output_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
            raise
            
    def plot_learning_curves(
        self,
        train_scores: List[float],
        val_scores: List[float],
        epochs: List[int],
        metric_name: str = 'Loss',
        output_path: Optional[str] = None
    ) -> go.Figure:
        """Plot learning curves showing training and validation metrics."""
        try:
            fig = go.Figure()
            
            # Add training scores
            fig.add_trace(go.Scatter(
                x=epochs,
                y=train_scores,
                mode='lines',
                name='Training',
                line=dict(color='blue')
            ))
            
            # Add validation scores
            fig.add_trace(go.Scatter(
                x=epochs,
                y=val_scores,
                mode='lines',
                name='Validation',
                line=dict(color='red')
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Learning Curves - {metric_name}',
                xaxis_title='Epoch',
                yaxis_title=metric_name,
                width=self.style_config['width'],
                height=self.style_config['height'],
                template=self.style_config['template']
            )
            
            if output_path:
                fig.write_html(output_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting learning curves: {str(e)}")
            raise
            
    def plot_residuals(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        output_path: Optional[str] = None
    ) -> go.Figure:
        """Plot residuals analysis for regression models."""
        try:
            residuals = actual - predicted
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Residuals vs Predicted',
                    'Residual Distribution',
                    'Q-Q Plot',
                    'Residuals vs Index'
                )
            )
            
            # Residuals vs Predicted
            fig.add_trace(
                go.Scatter(
                    x=predicted,
                    y=residuals,
                    mode='markers',
                    name='Residuals'
                ),
                row=1, col=1
            )
            
            # Residual Distribution
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name='Distribution'
                ),
                row=1, col=2
            )
            
            # Q-Q Plot
            theoretical_q = np.random.normal(
                np.mean(residuals),
                np.std(residuals),
                len(residuals)
            )
            theoretical_q.sort()
            residuals_sorted = np.sort(residuals)
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_q,
                    y=residuals_sorted,
                    mode='markers',
                    name='Q-Q Plot'
                ),
                row=2, col=1
            )
            
            # Residuals vs Index
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(residuals))),
                    y=residuals,
                    mode='markers',
                    name='vs Index'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=900,
                width=1200,
                showlegend=False,
                title_text="Residuals Analysis",
                template=self.style_config['template']
            )
            
            if output_path:
                fig.write_html(output_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting residuals: {str(e)}")
            raise
            
    def plot_correlation_matrix(
        self,
        correlation_matrix: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> go.Figure:
        """Plot correlation matrix heatmap."""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            # Update layout
            fig.update_layout(
                title='Feature Correlation Matrix',
                width=1000,
                height=1000,
                template=self.style_config['template']
            )
            
            if output_path:
                fig.write_html(output_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {str(e)}")
            raise
            
    def plot_metric_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        output_path: Optional[str] = None
    ) -> go.Figure:
        """Plot comparison of different metrics across models."""
        try:
            # Prepare data
            models = list(metrics.keys())
            metric_names = list(metrics[models[0]].keys())
            
            # Create grouped bar plot
            fig = go.Figure()
            
            for metric in metric_names:
                values = [metrics[model][metric] for model in models]
                fig.add_trace(go.Bar(
                    name=metric,
                    x=models,
                    y=values
                ))
                
            # Update layout
            fig.update_layout(
                title='Model Metrics Comparison',
                xaxis_title='Models',
                yaxis_title='Score',
                barmode='group',
                width=self.style_config['width'],
                height=self.style_config['height'],
                template=self.style_config['template']
            )
            
            if output_path:
                fig.write_html(output_path)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting metric comparison: {str(e)}")
            raise
            
    def update_style_config(self, new_config: Dict):
        """Update visualization style configuration."""
        self.style_config.update(new_config)