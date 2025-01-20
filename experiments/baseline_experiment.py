"""
Baseline model experiments for hotel booking analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import logging
from datetime import datetime
import json

from src.data_processor import DataProcessor
from src.utils import setup_logger
from src.config import CONFIG
from evaluation.model_evaluator import ModelEvaluator
from evaluation.performance_tracker import PerformanceTracker
from evaluation.visualization import VisualizationManager

logger = setup_logger(__name__)

class BaselineExperiment:
    """Runs baseline model experiments for hotel booking prediction."""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_evaluator = ModelEvaluator()
        self.performance_tracker = PerformanceTracker()
        self.viz_manager = VisualizationManager()
        
        # Initialize baseline models
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=CONFIG['RANDOM_STATE']
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=CONFIG['RANDOM_STATE']
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=CONFIG['RANDOM_STATE']
            )
        }
        
        self.results: Dict[str, Dict] = {}
        
    def run_experiments(
        self,
        data_path: str,
        output_dir: str
    ):
        """Run all baseline experiments."""
        try:
            logger.info("Starting baseline experiments")
            
            # Load and preprocess data
            df = self.data_processor.load_data(data_path)
            df = self.data_processor.preprocess_data(df)
            
            # Prepare features and target
            features, target = self.data_processor.prepare_features_targets(
                df,
                target_col='is_canceled'
            )
            
            # Split data
            X_train, X_test, y_train, y_test = self._split_data(features, target)
            
            # Run experiments for each model
            for model_name, model in self.models.items():
                logger.info(f"Running experiment for {model_name}")
                
                # Train and evaluate model
                model_results = self._run_single_experiment(
                    model,
                    model_name,
                    X_train,
                    X_test,
                    y_train,
                    y_test
                )
                
                self.results[model_name] = model_results
                
            # Generate comparison visualizations
            self._generate_visualizations(output_dir)
            
            # Save results
            self._save_results(output_dir)
            
            logger.info("Baseline experiments completed successfully")
            
        except Exception as e:
            logger.error(f"Error in baseline experiments: {str(e)}")
            raise
            
    def _split_data(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=CONFIG['TEST_SIZE'],
            random_state=CONFIG['RANDOM_STATE']
        )
        
        return X_train, X_test, y_train, y_test
        
    def _run_single_experiment(
        self,
        model: any,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """Run experiment for a single model."""
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Evaluate model
            metrics = self.model_evaluator.evaluate_classifier(
                y_test,
                y_pred,
                y_prob
            )
            
            # Calculate feature importance if available
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    X_train.columns,
                    model.feature_importances_
                ))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(
                    X_train.columns,
                    abs(model.coef_[0])
                ))
                
            # Cross-validation scores
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring='accuracy'
            )
            
            # Track performance
            self.performance_tracker.track_performance(
                model_name,
                y_prob,
                y_test,
                latency=0.0,  # Placeholder
                memory_usage=0.0  # Placeholder
            )
            
            return {
                'model': model,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'cv_scores': {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                },
                'predictions': {
                    'y_pred': y_pred,
                    'y_prob': y_prob
                }
            }
            
        except Exception as e:
            logger.error(f"Error in single experiment: {str(e)}")
            raise
            
    def _generate_visualizations(self, output_dir: str):
        """Generate visualization for experiment results."""
        try:
            # Comparison of model metrics
            metrics_comparison = {
                model_name: {
                    'accuracy': results['metrics'].accuracy,
                    'precision': results['metrics'].precision,
                    'recall': results['metrics'].recall,
                    'f1_score': results['metrics'].f1_score
                }
                for model_name, results in self.results.items()
            }
            
            self.viz_manager.plot_metric_comparison(
                metrics_comparison,
                output_path=f"{output_dir}/baseline_metrics_comparison.html"
            )
            
            # Feature importance comparison
            for model_name, results in self.results.items():
                if results['feature_importance']:
                    self.viz_manager.plot_feature_importance(
                        results['feature_importance'],
                        output_path=f"{output_dir}/{model_name}_feature_importance.html"
                    )
                    
            # ROC curves
            for model_name, results in self.results.items():
                y_test = results['predictions']['y_test']
                y_prob = results['predictions']['y_prob']
                
                self.viz_manager.plot_roc_curve(
                    y_test,
                    y_prob[:, 1],
                    results['metrics'].auc_roc,
                    output_path=f"{output_dir}/{model_name}_roc_curve.html"
                )
                
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise
            
    def _save_results(self, output_dir: str):
        """Save experiment results."""
        try:
            results_summary = {
                model_name: {
                    'metrics': {
                        'accuracy': results['metrics'].accuracy,
                        'precision': results['metrics'].precision,
                        'recall': results['metrics'].recall,
                        'f1_score': results['metrics'].f1_score,
                        'auc_roc': results['metrics'].auc_roc
                    },
                    'cv_scores': results['cv_scores'],
                    'feature_importance': results['feature_importance']
                }
                for model_name, results in self.results.items()
            }
            
            # Save to JSON
            with open(f"{output_dir}/baseline_results.json", 'w') as f:
                json.dump(results_summary, f, indent=4)
                
            logger.info(f"Results saved to {output_dir}/baseline_results.json")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise