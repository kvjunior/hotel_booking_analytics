"""
Model comparison experiments for hotel booking analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import setup_logger
from src.config import CONFIG
from evaluation.visualization import VisualizationManager

logger = setup_logger(__name__)

class ModelComparison:
    """Compares results from baseline and advanced experiments."""
    
    def __init__(self):
        self.viz_manager = VisualizationManager()
        self.comparison_results: Dict = {}
        
    def compare_models(
        self,
        baseline_results_path: str,
        advanced_results_path: str,
        output_dir: str
    ):
        """Compare baseline and advanced model results."""
        try:
            logger.info("Starting model comparison")
            
            # Load results
            baseline_results = self._load_results(baseline_results_path)
            advanced_results = self._load_results(advanced_results_path)
            
            # Compare metrics
            metrics_comparison = self._compare_metrics(
                baseline_results,
                advanced_results
            )
            
            # Statistical significance tests
            significance_tests = self._run_significance_tests(
                baseline_results,
                advanced_results
            )
            
            # Feature importance comparison
            feature_importance_comparison = self._compare_feature_importance(
                baseline_results,
                advanced_results
            )
            
            # Generate comparison visualizations
            self._generate_comparison_visualizations(
                metrics_comparison,
                feature_importance_comparison,
                output_dir
            )
            
            # Store comparison results
            self.comparison_results = {
                'metrics_comparison': metrics_comparison,
                'significance_tests': significance_tests,
                'feature_importance_comparison': feature_importance_comparison
            }
            
            # Save comparison results
            self._save_comparison_results(output_dir)
            
            logger.info("Model comparison completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model comparison: {str(e)}")
            raise
            
    def _load_results(self, results_path: str) -> Dict:
        """Load experiment results from JSON file."""
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            logger.error(f"Error loading results from {results_path}: {str(e)}")
            raise
            
    def _compare_metrics(
        self,
        baseline_results: Dict,
        advanced_results: Dict
    ) -> Dict:
        """Compare performance metrics between baseline and advanced models."""
        try:
            metrics_comparison = {}
            
            # Get all unique metrics
            all_metrics = set()
            for results in [baseline_results, advanced_results]:
                for model in results.values():
                    all_metrics.update(model['metrics'].keys())
                    
            # Compare each metric
            for metric in all_metrics:
                baseline_scores = {
                    model: results['metrics'][metric]
                    for model, results in baseline_results.items()
                    if metric in results['metrics']
                }
                
                advanced_scores = {
                    model: results['metrics'][metric]
                    for model, results in advanced_results.items()
                    if metric in results['metrics']
                }
                
                metrics_comparison[metric] = {
                    'baseline': baseline_scores,
                    'advanced': advanced_scores,
                    'improvement': {
                        model: advanced_scores[model] - baseline_scores.get(model, 0)
                        for model in advanced_scores
                    }
                }
                
            return metrics_comparison
            
        except Exception as e:
            logger.error(f"Error comparing metrics: {str(e)}")
            raise
            
    def _run_significance_tests(
        self,
        baseline_results: Dict,
        advanced_results: Dict
    ) -> Dict:
        """Run statistical significance tests on model predictions."""
        try:
            significance_tests = {}
            
            for model in advanced_results.keys():
                if model in baseline_results:
                    # Get predictions from both models
                    baseline_pred = baseline_results[model]['predictions']['y_pred']
                    advanced_pred = advanced_results[model]['predictions']['y_pred']
                    
                    # McNemar's test for paired nominal data
                    contingency_table = pd.crosstab(
                        baseline_pred,
                        advanced_pred
                    )
                    
                    statistic, p_value = stats.mcnemar(contingency_table)
                    
                    significance_tests[model] = {
                        'test_name': "McNemar's Test",
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
                    
            return significance_tests
            
        except Exception as e:
            logger.error(f"Error running significance tests: {str(e)}")
            raise
            
    def _compare_feature_importance(
        self,
        baseline_results: Dict,
        advanced_results: Dict
    ) -> Dict:
        """Compare feature importance between baseline and advanced models."""
        try:
            feature_comparison = {}
            
            for model in advanced_results.keys():
                if model in baseline_results:
                    baseline_importance = baseline_results[model].get('feature_importance', {})
                    advanced_importance = advanced_results[model].get('feature_importance', {})
                    
                    if baseline_importance and advanced_importance:
                        # Get all features
                        all_features = set(baseline_importance.keys()) | \
                                     set(advanced_importance.keys())
                                     
                        feature_comparison[model] = {
                            feature: {
                                'baseline': baseline_importance.get(feature, 0),
                                'advanced': advanced_importance.get(feature, 0),
                                'difference': advanced_importance.get(feature, 0) - \
                                            baseline_importance.get(feature, 0)
                            }
                            for feature in all_features
                        }
                        
            return feature_comparison
            
        except Exception as e:
            logger.error(f"Error comparing feature importance: {str(e)}")
            raise
            
    def _generate_comparison_visualizations(
        self,
        metrics_comparison: Dict,
        feature_importance_comparison: Dict,
        output_dir: str
    ):
        """Generate visualizations comparing model results."""
        try:
            # Metrics comparison plot
            fig = make_subplots(
                rows=len(metrics_comparison),
                cols=1,
                subplot_titles=list(metrics_comparison.keys())
            )
            
            for i, (metric, comparison) in enumerate(metrics_comparison.items(), 1):
                baseline_values = list(comparison['baseline'].values())
                advanced_values = list(comparison['advanced'].values())
                models = list(comparison['baseline'].keys())
                
                fig.add_trace(
                    go.Bar(
                        name='Baseline',
                        x=models,
                        y=baseline_values,
                        offsetgroup=0
                    ),
                    row=i, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        name='Advanced',
                        x=models,
                        y=advanced_values,
                        offsetgroup=1
                    ),
                    row=i, col=1
                )
                
            fig.update_layout(
                height=300*len(metrics_comparison),
                title_text="Model Performance Comparison",
                barmode='group'
            )
            
            fig.write_html(f"{output_dir}/metrics_comparison.html")
            
            # Feature importance comparison
            for model, importance_comparison in feature_importance_comparison.items():
                features = list(importance_comparison.keys())
                baseline_importance = [comp['baseline'] for comp in importance_comparison.values()]
                advanced_importance = [comp['advanced'] for comp in importance_comparison.values()]
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Baseline',
                    x=features,
                    y=baseline_importance,
                    offsetgroup=0
                ))
                
                fig.add_trace(go.Bar(
                    name='Advanced',
                    x=features,
                    y=advanced_importance,
                    offsetgroup=1
                ))
                
                fig.update_layout(
                    title=f'Feature Importance Comparison - {model}',
                    xaxis_tickangle=-45,
                    barmode='group'
                )
                
                fig.write_html(f"{output_dir}/feature_importance_comparison_{model}.html")
                
        except Exception as e:
            logger.error(f"Error generating comparison visualizations: {str(e)}")
            raise
            
    def _save_comparison_results(self, output_dir: str):
        """Save comparison results to JSON."""
        try:
            with open(f"{output_dir}/comparison_results.json", 'w') as f:
                json.dump(self.comparison_results, f, indent=4)
                
            logger.info(f"Comparison results saved to {output_dir}/comparison_results.json")
            
        except Exception as e:
            logger.error(f"Error saving comparison results: {str(e)}")
            raise
            
    def get_best_model(self) -> Tuple[str, Dict]:
        """Identify the best performing model based on metrics."""
        try:
            if not self.comparison_results:
                raise ValueError("No comparison results available")
                
            # Get accuracy comparison
            accuracy_comparison = self.comparison_results['metrics_comparison']['accuracy']
            
            # Combine baseline and advanced results
            all_accuracies = {
                f"baseline_{model}": acc
                for model, acc in accuracy_comparison['baseline'].items()
            }
            all_accuracies.update({
                f"advanced_{model}": acc
                for model, acc in accuracy_comparison['advanced'].items()
            })
            
            # Find best model
            best_model = max(all_accuracies.items(), key=lambda x: x[1])
            
            return best_model[0], {
                'accuracy': best_model[1],
                'significant_improvement': self.comparison_results['significance_tests'] \
                    .get(best_model[0], {}).get('significant', False)
            }
            
        except Exception as e:
            logger.error(f"Error identifying best model: {str(e)}")
            raise