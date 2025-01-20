"""
Adaptive Hotel Optimization System implementing dynamic learning and regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

from ..config import CONFIG
from ..utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class MarketState:
    """Current market state representation."""
    demand_level: float
    price_sensitivity: float
    competition_intensity: float
    seasonality_factor: float
    occupancy_rate: float

@dataclass
class OptimizationResult:
    """Optimization result container."""
    optimal_parameters: Dict[str, float]
    expected_revenue: float
    confidence_score: float
    adaptation_metrics: Dict[str, float]

class AdaptiveHotelOptimizer:
    """Dynamic hotel optimization system with adaptive learning capabilities."""
    
    def __init__(self):
        self._initialize_models()
        self._initialize_market_trackers()
        self.scaler = StandardScaler()
        self.current_state: Optional[MarketState] = None
        
    def _initialize_models(self):
        """Initialize all optimization models."""
        # Price prediction model
        self.price_model = RandomForestRegressor(
            n_estimators=CONFIG['PREDICTION']['RF_N_ESTIMATORS'],
            max_depth=CONFIG['PREDICTION']['RF_MAX_DEPTH'],
            random_state=CONFIG['RANDOM_STATE']
        )
        
        # Demand prediction model
        self.demand_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(None, 10)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.demand_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Market regime classifier
        self.regime_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=CONFIG['RANDOM_STATE']
        )
        
    def _initialize_market_trackers(self):
        """Initialize market tracking mechanisms."""
        self.market_history = {
            'demand_levels': [],
            'price_trends': [],
            'occupancy_rates': [],
            'revenue_metrics': []
        }
        
        self.adaptation_metrics = {
            'learning_rate': 0.01,
            'exploration_rate': 0.1,
            'confidence_threshold': 0.8
        }
        
    def update_market_state(
        self,
        new_bookings: pd.DataFrame,
        competitor_data: Optional[pd.DataFrame] = None
    ) -> MarketState:
        """Update current market state based on new data."""
        try:
            # Calculate demand level
            demand_level = self._calculate_demand_level(new_bookings)
            
            # Calculate price sensitivity
            price_sensitivity = self._calculate_price_sensitivity(new_bookings)
            
            # Analyze competition if data available
            competition_intensity = self._analyze_competition(competitor_data) \
                if competitor_data is not None else 0.5
                
            # Calculate seasonality
            seasonality_factor = self._calculate_seasonality(new_bookings)
            
            # Calculate occupancy
            occupancy_rate = self._calculate_occupancy_rate(new_bookings)
            
            # Update current state
            self.current_state = MarketState(
                demand_level=demand_level,
                price_sensitivity=price_sensitivity,
                competition_intensity=competition_intensity,
                seasonality_factor=seasonality_factor,
                occupancy_rate=occupancy_rate
            )
            
            self._update_market_history()
            
            logger.info("Market state updated successfully")
            return self.current_state
            
        except Exception as e:
            logger.error(f"Error updating market state: {str(e)}")
            raise
            
    def optimize_parameters(
        self,
        current_bookings: pd.DataFrame,
        constraints: Dict[str, any]
    ) -> OptimizationResult:
        """Optimize hotel parameters based on current state and constraints."""
        try:
            if not self.current_state:
                raise ValueError("Market state not initialized")
                
            # Prepare optimization inputs
            features = self._prepare_optimization_features(
                current_bookings,
                constraints
            )
            
            # Optimize pricing
            optimal_price = self._optimize_pricing(features)
            
            # Optimize capacity allocation
            optimal_capacity = self._optimize_capacity(features, optimal_price)
            
            # Optimize service levels
            optimal_service = self._optimize_service_levels(
                features,
                optimal_price,
                optimal_capacity
            )
            
            # Combine optimal parameters
            optimal_parameters = {
                'pricing': optimal_price,
                'capacity': optimal_capacity,
                'service_levels': optimal_service
            }
            
            # Calculate expected revenue
            expected_revenue = self._calculate_expected_revenue(
                optimal_parameters,
                features
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                optimal_parameters,
                features
            )
            
            # Update adaptation metrics
            self._update_adaptation_metrics(
                optimal_parameters,
                expected_revenue,
                confidence_score
            )
            
            result = OptimizationResult(
                optimal_parameters=optimal_parameters,
                expected_revenue=expected_revenue,
                confidence_score=confidence_score,
                adaptation_metrics=self.adaptation_metrics
            )
            
            logger.info("Parameter optimization completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {str(e)}")
            raise
            
    def _calculate_demand_level(self, bookings: pd.DataFrame) -> float:
        """Calculate current demand level."""
        recent_bookings = bookings.sort_values('arrival_date').tail(30)
        total_capacity = len(recent_bookings) * 100  # Assuming 100 rooms
        actual_bookings = len(recent_bookings[recent_bookings['is_canceled'] == 0])
        return actual_bookings / total_capacity
        
    def _calculate_price_sensitivity(self, bookings: pd.DataFrame) -> float:
        """Calculate price sensitivity index."""
        try:
            # Group by price ranges and calculate booking rates
            bookings['price_range'] = pd.qcut(bookings['adr'], q=5)
            sensitivity = bookings.groupby('price_range')['is_canceled'].mean()
            return float(sensitivity.std())  # Higher std = higher sensitivity
        except Exception:
            return 0.5  # Default value
            
    def _analyze_competition(self, competitor_data: pd.DataFrame) -> float:
        """Analyze competition intensity."""
        if competitor_data is None or competitor_data.empty:
            return 0.5
            
        our_prices = competitor_data['our_price'].mean()
        comp_prices = competitor_data['competitor_price'].mean()
        
        return float(np.clip(comp_prices / our_prices, 0, 1))
        
    def _calculate_seasonality(self, bookings: pd.DataFrame) -> float:
        """Calculate seasonality factor."""
        monthly_bookings = bookings.groupby('arrival_date_month')['adr'].mean()
        return float(monthly_bookings.std() / monthly_bookings.mean())
        
    def _calculate_occupancy_rate(self, bookings: pd.DataFrame) -> float:
        """Calculate current occupancy rate."""
        recent_bookings = bookings[bookings['is_canceled'] == 0].tail(30)
        return len(recent_bookings) / (30 * 100)  # Assuming 100 rooms
        
    def _prepare_optimization_features(
        self,
        bookings: pd.DataFrame,
        constraints: Dict[str, any]
    ) -> pd.DataFrame:
        """Prepare features for optimization."""
        features = pd.DataFrame()
        
        # Add market state features
        features['demand_level'] = [self.current_state.demand_level]
        features['price_sensitivity'] = [self.current_state.price_sensitivity]
        features['competition_intensity'] = [self.current_state.competition_intensity]
        features['seasonality_factor'] = [self.current_state.seasonality_factor]
        features['occupancy_rate'] = [self.current_state.occupancy_rate]
        
        # Add constraint features
        for key, value in constraints.items():
            features[f'constraint_{key}'] = [value]
            
        return features
        
    def _optimize_pricing(self, features: pd.DataFrame) -> Dict[str, float]:
        """Optimize pricing strategy."""
        base_price = 100  # Base price point
        
        # Adjust for demand
        demand_multiplier = 1 + (features['demand_level'].iloc[0] - 0.5)
        
        # Adjust for competition
        competition_multiplier = 1 - (features['competition_intensity'].iloc[0] - 0.5)
        
        # Adjust for seasonality
        season_multiplier = 1 + features['seasonality_factor'].iloc[0]
        
        optimal_price = base_price * demand_multiplier * \
                       competition_multiplier * season_multiplier
                       
        return {
            'base_rate': float(optimal_price),
            'weekend_multiplier': 1.2 if features['demand_level'].iloc[0] > 0.7 else 1.1,
            'extended_stay_discount': 0.1 if features['occupancy_rate'].iloc[0] < 0.6 else 0.05
        }
        
    def _optimize_capacity(
        self,
        features: pd.DataFrame,
        optimal_price: Dict[str, float]
    ) -> Dict[str, float]:
        """Optimize capacity allocation."""
        return {
            'standard_rooms': 0.6,
            'premium_rooms': 0.3,
            'suite_rooms': 0.1
        }
        
    def _optimize_service_levels(
        self,
        features: pd.DataFrame,
        optimal_price: Dict[str, float],
        optimal_capacity: Dict[str, float]
    ) -> Dict[str, float]:
        """Optimize service levels."""
        return {
            'cleaning_frequency': 1.5 if features['occupancy_rate'].iloc[0] > 0.8 else 1.0,
            'staff_ratio': 0.15 if features['demand_level'].iloc[0] > 0.7 else 0.1,
            'amenity_level': 2.0 if optimal_price['base_rate'] > 150 else 1.5
        }
        
    def _calculate_expected_revenue(
        self,
        optimal_parameters: Dict[str, any],
        features: pd.DataFrame
    ) -> float:
        """Calculate expected revenue from optimization."""
        base_revenue = optimal_parameters['pricing']['base_rate'] * \
                      features['demand_level'].iloc[0] * 100  # Assuming 100 rooms
                      
        # Adjust for capacity mix
        capacity_multiplier = sum(
            rate * share for rate, share in 
            zip([1.0, 1.3, 1.8], optimal_parameters['capacity'].values())
        )
        
        return float(base_revenue * capacity_multiplier)
        
    def _calculate_confidence_score(
        self,
        optimal_parameters: Dict[str, any],
        features: pd.DataFrame
    ) -> float:
        """Calculate confidence score for optimization results."""
        base_confidence = 0.8
        
        # Reduce confidence for extreme values
        if features['demand_level'].iloc[0] < 0.2 or \
           features['demand_level'].iloc[0] > 0.8:
            base_confidence *= 0.9
            
        # Reduce confidence for high competition
        if features['competition_intensity'].iloc[0] > 0.7:
            base_confidence *= 0.9
            
        # Reduce confidence for high seasonality
        if features['seasonality_factor'].iloc[0] > 0.5:
            base_confidence *= 0.9
            
        return float(base_confidence)
        
    def _update_adaptation_metrics(
        self,
        optimal_parameters: Dict[str, any],
        expected_revenue: float,
        confidence_score: float
    ):
        """Update adaptation metrics based on optimization results."""
        # Update learning rate based on confidence
        self.adaptation_metrics['learning_rate'] *= \
            (0.9 + 0.2 * confidence_score)
            
        # Update exploration rate inversely to confidence
        self.adaptation_metrics['exploration_rate'] = \
            0.1 * (1 - confidence_score)
            
        # Update confidence threshold
        self.adaptation_metrics['confidence_threshold'] = \
            min(0.9, self.adaptation_metrics['confidence_threshold'] * 1.01)
            
    def _update_market_history(self):
        """Update market history with current state."""
        if self.current_state:
            self.market_history['demand_levels'].append(
                self.current_state.demand_level
            )
            self.market_history['occupancy_rates'].append(
                self.current_state.occupancy_rate
            )
            
            # Keep only recent history
            max_history = 90  # 90 days of history
            for key in self.market_history:
                if len(self.market_history[key]) > max_history:
                    self.market_history[key] = \
                        self.market_history[key][-max_history:]