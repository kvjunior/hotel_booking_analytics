"""
Booking prediction system using ensemble methods and deep learning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from .config import CONFIG
from .utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class PredictionResult:
    """Class for storing prediction results."""
    probability: float
    confidence: float
    feature_importance: Dict[str, float]
    model_contributions: Dict[str, float]

class BookingPredictor:
    """Predicts booking outcomes using ensemble of models."""
    
    def __init__(self):
        # Initialize base models
        self.rf_model = RandomForestClassifier(
            n_estimators=CONFIG['PREDICTION']['RF_N_ESTIMATORS'],
            max_depth=CONFIG['PREDICTION']['RF_MAX_DEPTH'],
            random_state=CONFIG['RANDOM_STATE']
        )
        
        self.xgb_model = xgb.XGBClassifier(
            learning_rate=CONFIG['PREDICTION']['XGB_LEARNING_RATE'],
            max_depth=CONFIG['PREDICTION']['XGB_MAX_DEPTH'],
            random_state=CONFIG['RANDOM_STATE']
        )
        
        self.gb_model = GradientBoostingClassifier(
            learning_rate=CONFIG['PREDICTION']['XGB_LEARNING_RATE'],
            max_depth=CONFIG['PREDICTION']['XGB_MAX_DEPTH'],
            random_state=CONFIG['RANDOM_STATE']
        )
        
        # Initialize neural network
        self.nn_model = self._build_neural_network()
        
        # Model weights for ensemble
        self.model_weights = {
            'rf': 0.3,
            'xgb': 0.3,
            'gb': 0.2,
            'nn': 0.2
        }
        
        self.feature_importance: Optional[Dict[str, float]] = None
        
    def _build_neural_network(self) -> Sequential:
        """Build and compile neural network model."""
        model = Sequential([
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(CONFIG['PREDICTION']['NN_DROPOUT_RATE']),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(CONFIG['PREDICTION']['NN_DROPOUT_RATE']),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=CONFIG['PREDICTION']['XGB_LEARNING_RATE'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ):
        """Train all models in the ensemble."""
        try:
            logger.info("Starting model training...")
            
            # Train Random Forest
            self.rf_model.fit(X_train, y_train)
            rf_score = self.rf_model.score(X_val, y_val)
            logger.info(f"Random Forest validation score: {rf_score:.4f}")
            
            # Train XGBoost
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            xgb_score = self.xgb_model.score(X_val, y_val)
            logger.info(f"XGBoost validation score: {xgb_score:.4f}")
            
            # Train Gradient Boosting
            self.gb_model.fit(X_train, y_train)
            gb_score = self.gb_model.score(X_val, y_val)
            logger.info(f"Gradient Boosting validation score: {gb_score:.4f}")
            
            # Train Neural Network
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            self.nn_model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            nn_score = self.nn_model.evaluate(X_val, y_val, verbose=0)[1]
            logger.info(f"Neural Network validation score: {nn_score:.4f}")
            
            # Update model weights based on performance
            total_score = rf_score + xgb_score + gb_score + nn_score
            self.model_weights = {
                'rf': rf_score / total_score,
                'xgb': xgb_score / total_score,
                'gb': gb_score / total_score,
                'nn': nn_score / total_score
            }
            
            # Calculate feature importance
            self._calculate_feature_importance(X_train.columns)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
            
    def predict(self, features: pd.DataFrame) -> List[PredictionResult]:
        """Make predictions using the ensemble."""
        try:
            # Get predictions from each model
            rf_pred = self.rf_model.predict_proba(features)[:, 1]
            xgb_pred = self.xgb_model.predict_proba(features)[:, 1]
            gb_pred = self.gb_model.predict_proba(features)[:, 1]
            nn_pred = self.nn_model.predict(features).flatten()
            
            results = []
            for i in range(len(features)):
                # Calculate weighted ensemble prediction
                model_contributions = {
                    'rf': rf_pred[i] * self.model_weights['rf'],
                    'xgb': xgb_pred[i] * self.model_weights['xgb'],
                    'gb': gb_pred[i] * self.model_weights['gb'],
                    'nn': nn_pred[i] * self.model_weights['nn']
                }
                
                ensemble_pred = sum(model_contributions.values())
                
                # Calculate prediction confidence
                predictions = [rf_pred[i], xgb_pred[i], gb_pred[i], nn_pred[i]]
                confidence = 1 - np.std(predictions)
                
                result = PredictionResult(
                    probability=ensemble_pred,
                    confidence=confidence,
                    feature_importance=self.feature_importance,
                    model_contributions=model_contributions
                )
                
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
            
    def _calculate_feature_importance(self, feature_names: List[str]):
        """Calculate and combine feature importance from all models."""
        try:
            # Get importance from Random Forest
            rf_importance = dict(zip(feature_names, 
                                   self.rf_model.feature_importances_))
            
            # Get importance from XGBoost
            xgb_importance = dict(zip(feature_names, 
                                    self.xgb_model.feature_importances_))
            
            # Get importance from Gradient Boosting
            gb_importance = dict(zip(feature_names, 
                                   self.gb_model.feature_importances_))
            
            # Combine importance scores using model weights
            self.feature_importance = {}
            for feature in feature_names:
                self.feature_importance[feature] = (
                    rf_importance[feature] * self.model_weights['rf'] +
                    xgb_importance[feature] * self.model_weights['xgb'] +
                    gb_importance[feature] * self.model_weights['gb']
                )
                
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            self.feature_importance = None
            
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            # Get predictions
            predictions = self.predict(X_test)
            y_pred = [1 if p.probability >= 0.5 else 0 for p in predictions]
            y_prob = [p.probability for p in predictions]
            
            # Calculate metrics
            classification_metrics = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )
            
            auc_score = roc_auc_score(y_test, y_prob)
            
            # Combine metrics
            metrics = {
                'accuracy': classification_metrics['accuracy'],
                'precision': classification_metrics['weighted avg']['precision'],
                'recall': classification_metrics['weighted avg']['recall'],
                'f1': classification_metrics['weighted avg']['f1-score'],
                'auc': auc_score
            }
            
            logger.info("Model evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
            
    def get_model_summary(self) -> Dict[str, any]:
        """Generate model summary report."""
        return {
            'model_weights': self.model_weights,
            'feature_importance': self.feature_importance,
            'model_parameters': {
                'rf_estimators': self.rf_model.n_estimators,
                'xgb_learning_rate': self.xgb_model.learning_rate,
                'gb_learning_rate': self.gb_model.learning_rate
            }
        }