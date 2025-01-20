"""
Data preprocessing and feature engineering for hotel booking analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime
import logging
from typing import Tuple, List, Dict, Optional

from .config import CONFIG
from .utils import setup_logger

logger = setup_logger(__name__)

class DataProcessor:
    """Handles all data preprocessing and feature engineering tasks."""
    
    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded data from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline."""
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Create date features
        df = self._create_date_features(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Handle categorical variables
        df = self._encode_categorical_features(df)
        
        # Scale numerical features
        df = self._scale_numerical_features(df)
        
        logger.info("Data preprocessing completed successfully")
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Numerical features
        for feature in CONFIG['NUMERICAL_FEATURES']:
            if feature in df.columns:
                self.imputers[feature] = SimpleImputer(strategy='median')
                df[feature] = self.imputers[feature].fit_transform(df[[feature]])
                
        # Categorical features
        for feature in CONFIG['CATEGORICAL_FEATURES']:
            if feature in df.columns:
                self.imputers[feature] = SimpleImputer(strategy='constant', fill_value='MISSING')
                df[feature] = self.imputers[feature].fit_transform(df[[feature]])
                
        return df
        
    def _create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from date-related columns."""
        # Create arrival date
        df['arrival_date'] = pd.to_datetime(
            df['arrival_date_year'].astype(str) + '-' +
            df['arrival_date_month'] + '-' +
            df['arrival_date_day_of_month'].astype(str)
        )
        
        # Extract useful date components
        df['arrival_day_of_week'] = df['arrival_date'].dt.dayofweek
        df['arrival_is_weekend'] = df['arrival_day_of_week'].isin([5, 6]).astype(int)
        df['arrival_season'] = df['arrival_date'].dt.quarter
        
        # Create holiday indicators (example)
        df['is_holiday_season'] = df['arrival_date_month'].isin(['December', 'July', 'August']).astype(int)
        
        return df
        
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        # Guest-related interactions
        df['total_guests'] = df['adults'] + df['children'] + df['babies']
        df['is_family'] = (df['children'] > 0) | (df['babies'] > 0)
        
        # Stay-related interactions
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df['total_cost'] = df['adr'] * df['total_nights']
        
        # Booking behavior interactions
        df['booking_changes_ratio'] = df['booking_changes'] / (df['lead_time'] + 1)
        df['previous_booking_ratio'] = (
            df['previous_bookings_not_canceled'] /
            (df['previous_cancellations'] + df['previous_bookings_not_canceled'] + 1)
        )
        
        return df
        
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        for feature in CONFIG['CATEGORICAL_FEATURES']:
            if feature in df.columns:
                self.encoders[feature] = LabelEncoder()
                df[feature] = self.encoders[feature].fit_transform(df[feature])
                
        return df
        
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        numerical_features = [f for f in CONFIG['NUMERICAL_FEATURES'] if f in df.columns]
        
        for feature in numerical_features:
            self.scalers[feature] = StandardScaler()
            df[feature] = self.scalers[feature].fit_transform(df[[feature]])
            
        return df
        
    def prepare_features_targets(
        self, 
        df: pd.DataFrame,
        target_col: str = 'is_canceled'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables for modeling."""
        # Remove unnecessary columns
        cols_to_drop = ['arrival_date'] + [target_col]
        features = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Get target
        target = df[target_col] if target_col in df.columns else None
        
        return features, target
        
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        feature_name: str
    ) -> np.ndarray:
        """Inverse transform scaled predictions."""
        if feature_name in self.scalers:
            return self.scalers[feature_name].inverse_transform(predictions.reshape(-1, 1))
        return predictions