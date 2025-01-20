"""
Advanced model experiments for hotel booking analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import xgboost as xgb
from lightgbm import LGBMClassifier
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.data_processor import DataProcessor
from src.utils import setup_logger
from src.config import CONFIG
from evaluation.model_evaluator import ModelEvaluator
from evaluation.performance_tracker import PerformanceTracker
from evaluation.visualization import VisualizationManager
from models.adaptive_optimizer import AdaptiveHotelOptimizer

logger = setup_logger(__name__)

class AdvancedExperiment:
    """Runs advanced model experiments with hyperparameter optimization."""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_evaluator = ModelEvaluator()
        self.performance_tracker = PerformanceTracker()
        self.viz_manager = VisualizationManager()
        self.adaptive_optimizer = AdaptiveHotelOptimizer()
        
        self.results: Dict[str, Dict] = {}
        
    def run_experiments(
        self,
        data_path: str,
        output_dir: str,
        n_trials: int = 50
    ):
        """Run all advanced experiments."""
        try:
            logger.info("Starting advanced experiments")
            
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
            
            # Run XGBoost experiment
            logger.info("Running XGBoost experiment")
            xgb_results = self._run_xgboost_experiment(
                X_train, X_test, y_train, y_test, n_trials
            )
            self.results['xgboost'] = xgb_results
            
            # Run LightGBM experiment
            logger.info("Running LightGBM experiment")
            lgb_results = self._run_lightgbm_experiment(
                X_train, X_test, y_train, y_test, n_trials
            )
            self.results['lightgbm'] = lgb_results
            
            # Run Neural Network experiment
            logger.info("Running Neural Network experiment")
            nn_results = self._run_neural_network_experiment(
                X_train, X_test, y_train, y_test, n_trials
            )
            self.results['neural_network'] = nn_results
            
            # Run Ensemble experiment
            logger.info("Running Ensemble experiment")
            ensemble_results = self._run_ensemble_experiment(
                X_train, X_test, y_train, y_test
            )
            self.results['ensemble'] = ensemble_results
            
            # Generate visualizations
            self._generate_visualizations(output_dir)
            
            # Save results
            self._save_results(output_dir)
            
            logger.info("Advanced experiments completed successfully")
            
        except Exception as e:
            logger.error(f"Error in advanced experiments: {str(e)}")
            raise
            
    def _run_xgboost_experiment(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        n_trials: int
    ) -> Dict:
        """Run XGBoost experiment with hyperparameter optimization."""
        try:
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0)
                }
                
                model = xgb.XGBClassifier(
                    **params,
                    random_state=CONFIG['RANDOM_STATE']
                )
                
                # Cross-validation score
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG['RANDOM_STATE'])
                scores = []
                
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_train_fold = X_train.iloc[train_idx]
                    y_train_fold = y_train.iloc[train_idx]
                    X_val_fold = X_train.iloc[val_idx]
                    y_val_fold = y_train.iloc[val_idx]
                    
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict_proba(X_val_fold)[:, 1]
                    score = roc_auc_score(y_val_fold, y_pred)
                    scores.append(score)
                    
                return np.mean(scores)
                
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            # Train final model with best parameters
            best_params = study.best_params
            final_model = xgb.XGBClassifier(
                **best_params,
                random_state=CONFIG['RANDOM_STATE']
            )
            final_model.fit(X_train, y_train)

            # Make predictions
            y_pred = final_model.predict(X_test)
            y_prob = final_model.predict_proba(X_test)

            # Evaluate model
            metrics = self.model_evaluator.evaluate_classifier(
                y_test,
                y_pred,
                y_prob
            )

            return {
                'model': final_model,
                'metrics': metrics,
                'feature_importance': dict(zip(
                    X_train.columns,
                    final_model.feature_importances_
                )),
                'best_params': best_params,
                'optimization_history': study.trials_dataframe(),
                'predictions': {
                    'y_pred': y_pred,
                    'y_prob': y_prob
                }
            }

        except Exception as e:
            logger.error(f"Error in XGBoost experiment: {str(e)}")
            raise

    def _run_lightgbm_experiment(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        n_trials: int
    ) -> Dict:
        """Run LightGBM experiment with hyperparameter optimization."""
        try:
            def objective(trial):
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
                }

                model = LGBMClassifier(
                    **params,
                    random_state=CONFIG['RANDOM_STATE']
                )

                # Cross-validation score
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG['RANDOM_STATE'])
                scores = []

                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_train_fold = X_train.iloc[train_idx]
                    y_train_fold = y_train.iloc[train_idx]
                    X_val_fold = X_train.iloc[val_idx]
                    y_val_fold = y_train.iloc[val_idx]

                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict_proba(X_val_fold)[:, 1]
                    score = roc_auc_score(y_val_fold, y_pred)
                    scores.append(score)

                return np.mean(scores)

            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            # Train final model with best parameters
            best_params = study.best_params
            final_model = LGBMClassifier(
                **best_params,
                random_state=CONFIG['RANDOM_STATE']
            )
            final_model.fit(X_train, y_train)

            # Make predictions
            y_pred = final_model.predict(X_test)
            y_prob = final_model.predict_proba(X_test)

            # Evaluate model
            metrics = self.model_evaluator.evaluate_classifier(
                y_test,
                y_pred,
                y_prob
            )

            return {
                'model': final_model,
                'metrics': metrics,
                'feature_importance': dict(zip(
                    X_train.columns,
                    final_model.feature_importances_
                )),
                'best_params': best_params,
                'optimization_history': study.trials_dataframe(),
                'predictions': {
                    'y_pred': y_pred,
                    'y_prob': y_prob
                }
            }

        except Exception as e:
            logger.error(f"Error in LightGBM experiment: {str(e)}")
            raise

    def _run_neural_network_experiment(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        n_trials: int
    ) -> Dict:
        """Run Neural Network experiment with hyperparameter optimization."""
        try:
            def create_model(trial):
                model = Sequential()
                
                # Input layer
                model.add(Dense(
                    trial.suggest_int('units_1', 32, 256),
                    activation=trial.suggest_categorical('activation_1', ['relu', 'elu']),
                    input_shape=(X_train.shape[1],)
                ))
                model.add(BatchNormalization())
                model.add(Dropout(trial.suggest_uniform('dropout_1', 0.1, 0.5)))

                # Hidden layers
                n_layers = trial.suggest_int('n_layers', 1, 3)
                for i in range(n_layers):
                    model.add(Dense(
                        trial.suggest_int(f'units_{i+2}', 16, 128),
                        activation=trial.suggest_categorical(f'activation_{i+2}', ['relu', 'elu'])
                    ))
                    model.add(BatchNormalization())
                    model.add(Dropout(trial.suggest_uniform(f'dropout_{i+2}', 0.1, 0.5)))

                # Output layer
                model.add(Dense(1, activation='sigmoid'))

                # Compile model
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
                    ),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )

                return model

            def objective(trial):
                model = create_model(trial)
                
                # Early stopping callback
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )

                # Train model
                history = model.fit(
                    X_train,
                    y_train,
                    epochs=100,
                    batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )

                # Evaluate on validation set
                val_score = model.evaluate(X_test, y_test, verbose=0)[1]
                return val_score

            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            # Train final model with best parameters
            final_model = create_model(study.best_trial)
            
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            # Train final model
            history = final_model.fit(
                X_train,
                y_train,
                epochs=100,
                batch_size=study.best_params['batch_size'],
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )

            # Make predictions
            y_pred = (final_model.predict(X_test) > 0.5).astype(int)
            y_prob = final_model.predict(X_test)

            # Evaluate model
            metrics = self.model_evaluator.evaluate_classifier(
                y_test,
                y_pred,
                y_prob
            )

            return {
                'model': final_model,
                'metrics': metrics,
                'feature_importance': None,  # Neural networks don't provide direct feature importance
                'best_params': study.best_params,
                'optimization_history': study.trials_dataframe(),
                'training_history': history.history,
                'predictions': {
                    'y_pred': y_pred,
                    'y_prob': y_prob
                }
            }

        except Exception as e:
            logger.error(f"Error in Neural Network experiment: {str(e)}")
            raise

    def _run_ensemble_experiment(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """Run ensemble experiment combining XGBoost, LightGBM, and Neural Network."""
        try:
            # Get predictions from all models
            xgb_pred = self.results['xgboost']['predictions']['y_prob']
            lgb_pred = self.results['lightgbm']['predictions']['y_prob']
            nn_pred = self.results['neural_network']['predictions']['y_prob']

            # Create ensemble predictions
            ensemble_pred_prob = (xgb_pred + lgb_pred + nn_pred) / 3
            ensemble_pred = (ensemble_pred_prob > 0.5).astype(int)

            # Evaluate ensemble
            metrics = self.model_evaluator.evaluate_classifier(
                y_test,
                ensemble_pred,
                ensemble_pred_prob
            )

            return {
                'metrics': metrics,
                'predictions': {
                    'y_pred': ensemble_pred,
                    'y_prob': ensemble_pred_prob
                },
                'model_weights': {
                    'xgboost': 1/3,
                    'lightgbm': 1/3,
                    'neural_network': 1/3
                }
            }

        except Exception as e:
            logger.error(f"Error in ensemble experiment: {str(e)}")
            raise

    def _generate_visualizations(self, output_dir: str):
        """Generate visualization for experiment results."""
        try:
            # Learning curves for Neural Network
            if 'neural_network' in self.results:
                history = self.results['neural_network']['training_history']
                self.viz_manager.plot_learning_curves(
                    history['accuracy'],
                    history['val_accuracy'],
                    list(range(len(history['accuracy']))),
                    'Accuracy',
                    output_path=f"{output_dir}/nn_learning_curves.html"
                )

            # Feature importance comparison
            feature_importance_models = {
                name: results for name, results in self.results.items()
                if results.get('feature_importance') is not None
            }

            if feature_importance_models:
                for model_name, results in feature_importance_models.items():
                    self.viz_manager.plot_feature_importance(
                        results['feature_importance'],
                        output_path=f"{output_dir}/{model_name}_feature_importance.html"
                    )

            # ROC curves comparison
            for model_name, results in self.results.items():
                self.viz_manager.plot_roc_curve(
                    y_test,
                    results['predictions']['y_prob'],
                    results['metrics'].auc_roc,
                    output_path=f"{output_dir}/{model_name}_roc_curve.html"
                )

            # Optimization history plots
            for model_name in ['xgboost', 'lightgbm', 'neural_network']:
                if model_name in self.results:
                    history_df = self.results[model_name]['optimization_history']
                    self.viz_manager.plot_optimization_history(
                        history_df,
                        output_path=f"{output_dir}/{model_name}_optimization_history.html"
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
                    'best_params': results.get('best_params'),
                    'feature_importance': results.get('feature_importance')
                }
                for model_name, results in self.results.items()
            }

            # Save to JSON
            with open(f"{output_dir}/advanced_results.json", 'w') as f:
                json.dump(results_summary, f, indent=4)

            logger.info(f"Results saved to {output_dir}/advanced_results.json")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise