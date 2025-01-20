"""
Adaptive pricing system with dynamic learning and real-time price optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import statsmodels.api as sm
from scipy.optimize import minimize

from ..config import CONFIG
from ..utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class PricePoint:
    """Price point with confidence and supporting metrics."""
    base_price: float
    adjustments: Dict[str, float]
    final_price: float
    confidence_score: float
    expected_demand: float
    price_elasticity: float
    competitive_position: float

@dataclass
class MarketCondition:
    """Current market condition assessment."""
    demand_level: float
    competition_intensity: float
    seasonality_factor: float
    special_events: List[str]
    price_sensitivity: float
    occupancy_rate: float

class AdaptivePricingSystem:
    """Advanced pricing system with real-time adaptation capabilities."""
    
    def __init__(self):
        self._initialize_models()
        self.scaler = StandardScaler()
        self.price_history: List[PricePoint] = []
        self.market_history: List[MarketCondition] = []
        self.learning_rate = CONFIG['PRICING']['LEARNING_RATE']
        
    def _initialize_models(self):
        """Initialize prediction and optimization models."""
        # Demand prediction model
        self.demand_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(None, 10)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        self.demand_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Price optimization model
        self.price_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=CONFIG['RANDOM_STATE']
        )
        
        # Elasticity estimation model
        self.elasticity_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=CONFIG['RANDOM_STATE']
        )
        
        # Market condition model
        self.market_model = Sequential([
            Dense(32, activation='relu', input_shape=(10,)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(5, activation='linear')
        ])
        self.market_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
    def optimize_price(
        self,
        booking_features: pd.Series,
        market_condition: MarketCondition,
        constraints: Optional[Dict[str, float]] = None
    ) -> PricePoint:
        """Optimize price based on current conditions and constraints."""
        try:
            # Calculate base price
            base_price = self._calculate_base_price(booking_features)
            
            # Calculate price adjustments
            adjustments = self._calculate_adjustments(
                base_price,
                booking_features,
                market_condition
            )
            
            # Apply constraints
            final_price = self._apply_pricing_constraints(
                base_price,
                adjustments,
                constraints
            )
            
            # Calculate confidence and supporting metrics
            confidence_score = self._calculate_confidence_score(
                final_price,
                market_condition
            )
            
            expected_demand = self._predict_demand(
                final_price,
                booking_features,
                market_condition
            )
            
            price_elasticity = self._estimate_price_elasticity(
                final_price,
                booking_features
            )
            
            competitive_position = self._assess_competitive_position(
                final_price,
                market_condition
            )
            
            # Create price point
            price_point = PricePoint(
                base_price=base_price,
                adjustments=adjustments,
                final_price=final_price,
                confidence_score=confidence_score,
                expected_demand=expected_demand,
                price_elasticity=price_elasticity,
                competitive_position=competitive_position
            )
            
            # Update price history
            self.price_history.append(price_point)
            
            logger.info(f"Optimized price: {final_price:.2f} (confidence: {confidence_score:.2f})")
            return price_point
            
        except Exception as e:
            logger.error(f"Error in price optimization: {str(e)}")
            raise
            
    def _calculate_base_price(self, features: pd.Series) -> float:
        """Calculate base price using historical data and booking features."""
        try:
            # Extract key features for base price calculation
            room_type = features['reserved_room_type']
            stay_length = features['stays_in_weekend_nights'] + \
                         features['stays_in_week_nights']
            lead_time = features['lead_time']
            
            # Apply base pricing rules
            base_price = 100  # Default base price
            
            # Room type adjustments
            room_multipliers = {
                'A': 1.5, 'B': 1.3, 'C': 1.1,
                'D': 1.0, 'E': 0.9, 'F': 0.8
            }
            base_price *= room_multipliers.get(room_type, 1.0)
            
            # Stay length adjustments
            if stay_length > 7:
                base_price *= 0.9  # Long stay discount
            elif stay_length < 2:
                base_price *= 1.1  # Short stay premium
                
            # Lead time adjustments
            if lead_time > 90:
                base_price *= 0.95  # Early booking discount
            elif lead_time < 7:
                base_price *= 1.15  # Last minute premium
                
            return float(base_price)
            
        except Exception as e:
            logger.error(f"Error calculating base price: {str(e)}")
            raise
            
    def _calculate_adjustments(
        self,
        base_price: float,
        features: pd.Series,
        market_condition: MarketCondition
    ) -> Dict[str, float]:
        """Calculate price adjustments based on various factors."""
        adjustments = {}
        
        # Demand-based adjustment
        adjustments['demand'] = self._calculate_demand_adjustment(
            base_price,
            market_condition.demand_level
        )
        
        # Seasonality adjustment
        adjustments['seasonality'] = self._calculate_seasonality_adjustment(
            base_price,
            market_condition.seasonality_factor
        )
        
        # Competition adjustment
        adjustments['competition'] = self._calculate_competition_adjustment(
            base_price,
            market_condition.competition_intensity
        )
        
        # Special events adjustment
        adjustments['events'] = self._calculate_events_adjustment(
            base_price,
            market_condition.special_events
        )
        
        # Occupancy adjustment
        adjustments['occupancy'] = self._calculate_occupancy_adjustment(
            base_price,
            market_condition.occupancy_rate
        )
        
        return adjustments
        
    def _calculate_demand_adjustment(
        self,
        base_price: float,
        demand_level: float
    ) -> float:
        """Calculate price adjustment based on demand level."""
        if demand_level > 0.8:
            return base_price * 0.2  # Increase price by up to 20%
        elif demand_level < 0.3:
            return -base_price * 0.15  # Decrease price by up to 15%
        else:
            return base_price * (demand_level - 0.5) * 0.2
            
    def _calculate_seasonality_adjustment(
        self,
        base_price: float,
        seasonality_factor: float
    ) -> float:
        """Calculate price adjustment based on seasonality."""
        return base_price * (seasonality_factor - 0.5) * 0.15
        
    def _calculate_competition_adjustment(
        self,
        base_price: float,
        competition_intensity: float
    ) -> float:
        """Calculate price adjustment based on competition."""
        return -base_price * (competition_intensity - 0.5) * 0.1
        
    def _calculate_events_adjustment(
        self,
        base_price: float,
        special_events: List[str]
    ) -> float:
        """Calculate price adjustment for special events."""
        if not special_events:
            return 0.0
            
        # Define event multipliers
        event_multipliers = {
            'conference': 0.15,
            'holiday': 0.2,
            'local_event': 0.1,
            'peak_season': 0.25
        }
        
        # Calculate total adjustment
        total_adjustment = sum(
            event_multipliers.get(event, 0.05)
            for event in special_events
        )
        
        return base_price * total_adjustment
        
    def _calculate_occupancy_adjustment(
        self,
        base_price: float,
        occupancy_rate: float
    ) -> float:
        """Calculate price adjustment based on occupancy rate."""
        if occupancy_rate > 0.9:
            return base_price * 0.25  # High occupancy premium
        elif occupancy_rate < 0.4:
            return -base_price * 0.2  # Low occupancy discount
        else:
            return base_price * (occupancy_rate - 0.65) * 0.3
            
    def _apply_pricing_constraints(
        self,
        base_price: float,
        adjustments: Dict[str, float],
        constraints: Optional[Dict[str, float]] = None
    ) -> float:
        """Apply pricing constraints and calculate final price."""
        # Calculate total adjustments
        total_adjustment = sum(adjustments.values())
        
        # Calculate preliminary price
        preliminary_price = base_price + total_adjustment
        
        if constraints:
            # Apply minimum price constraint
            if 'min_price' in constraints:
                preliminary_price = max(
                    preliminary_price,
                    constraints['min_price']
                )
                
            # Apply maximum price constraint
            if 'max_price' in constraints:
                preliminary_price = min(
                    preliminary_price,
                    constraints['max_price']
                )
                
            # Apply maximum adjustment constraint
            if 'max_adjustment' in constraints:
                max_adj = constraints['max_adjustment'] * base_price
                total_adjustment = max(min(total_adjustment, max_adj), -max_adj)
                preliminary_price = base_price + total_adjustment
                
        return float(preliminary_price)
        
    def _calculate_confidence_score(
        self,
        price: float,
        market_condition: MarketCondition
    ) -> float:
        """Calculate confidence score for price recommendation."""
        base_confidence = 0.8
        
        # Adjust for extreme market conditions
        if market_condition.demand_level < 0.2 or \
           market_condition.demand_level > 0.8:
            base_confidence *= 0.9
            
        # Adjust for high competition
        if market_condition.competition_intensity > 0.7:
            base_confidence *= 0.9
            
        # Adjust for special events
        if market_condition.special_events:
            base_confidence *= 0.95
            
        # Adjust for extreme prices
        if price < 50 or price > 500:
            base_confidence *= 0.9
            
        return float(base_confidence)
        
    def _predict_demand(
        self,
        price: float,
        features: pd.Series,
        market_condition: MarketCondition
    ) -> float:
        """Predict demand for given price and conditions."""
        try:
            # Prepare features for demand prediction
            input_features = np.array([
                price,
                market_condition.demand_level,
                market_condition.seasonality_factor,
                market_condition.competition_intensity,
                market_condition.occupancy_rate,
                features['lead_time'],
                features['stays_in_weekend_nights'] + \
                features['stays_in_week_nights'],
                features['adults'] + features['children'],
                float(features['is_repeated_guest']),
                len(market_condition.special_events)
            ]).reshape(1, -1)
            
            # Scale features
            input_features_scaled = self.scaler.fit_transform(input_features)
            
            # Predict demand
            predicted_demand = float(self.demand_model.predict(
                input_features_scaled
            ))
            
            return max(min(predicted_demand, 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error predicting demand: {str(e)}")
            return 0.5  # Return moderate demand on error
            
    def _estimate_price_elasticity(
        self,
        price: float,
        features: pd.Series
    ) -> float:
        """Estimate price elasticity of demand."""
        try:
            # Use historical data to estimate elasticity
            if len(self.price_history) < 2:
                return -1.0  # Default elasticity
                
            # Calculate price and demand changes
            price_changes = [
                (p.final_price - prev_p.final_price) / prev_p.final_price
                for prev_p, p in zip(
                    self.price_history[:-1],
                    self.price_history[1:]
                )
            ]
            
            demand_changes = [
                (p.expected_demand - prev_p.expected_demand) / \
                prev_p.expected_demand
                for prev_p, p in zip(
                    self.price_history[:-1],
                    self.price_history[1:]
                )
            ]
            
            if not price_changes or not demand_changes:
                return -1.0
                
            # Calculate elasticity
            elasticity = np.mean([
                d_change / p_change
                for p_change, d_change in zip(price_changes, demand_changes)
                if abs(p_change) > 0.001  # Avoid division by very small numbers
            ])
            
            return float(elasticity)
            
        except Exception as e:
            logger.error(f"Error estimating price elasticity: {str(e)}")
            return -1.0
            
    def _assess_competitive_position(
        self,
        price: float,
        market_condition: MarketCondition
    ) -> float:
        """Assess competitive position for given price."""
        try:
            # Calculate relative price position
            if market_condition.competition_intensity > 0.8:
                # High competition - be more price sensitive
                if price > 100:
                    return 0.3  # Less competitive
                else:
                    return 0.7  # More competitive
            elif market_condition.demand_level > 0.7:
                # High demand - can be less price sensitive
                return 0.8
            else:
                # Normal conditions
                return 0.5
                
        except Exception as e:
            logger.error(f"Error assessing competitive position: {str(e)}")
            return 0.5
            
    def update_models(
        self,
        actual_bookings: pd.DataFrame,
        market_data: pd.DataFrame
    ):
        """Update prediction models with actual booking data."""
        try:
            # Prepare training data
            X_demand, y_demand = self._prepare_demand_training_data(
                actual_bookings
            )
            
            X_price, y_price = self._prepare_price_training_data(
                actual_bookings,
                market_data
            )
            
            # Update demand model
            self.demand_model.fit(
                X_demand,
                y_demand,
                epochs=5,
                batch_size=32,
                verbose=0
            )
            
            # Update price model
            self.price_model.fit(X_price, y_price)
            
            # Update elasticity model
            self._update_elasticity_model(actual_bookings)
            
            # Update market model
            self._update_market_model(market_data)
            
            logger.info("Successfully updated prediction models")
            
        except Exception as e:
            logger.error(f"Error updating models: {str(e)}")
            raise
            
    def _prepare_demand_training_data(
        self,
        bookings: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for demand model."""
        features = bookings[[
            'adr', 'lead_time', 'stays_in_weekend_nights',
            'stays_in_week_nights', 'adults', 'children',
            'is_repeated_guest', 'previous_cancellations',
            'previous_bookings_not_canceled', 'total_of_special_requests'
        ]].values
        
        targets = bookings['is_canceled'].values
        
        return self.scaler.fit_transform(features), targets
        
    def _prepare_price_training_data(
        self,
        bookings: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for price model."""
        features = pd.merge(
            bookings,
            market_data,
            on='arrival_date',
            how='left'
        )
        
        X = features[[
            'lead_time', 'total_of_special_requests',
            'market_segment', 'room_type', 'demand_level',
            'competition_intensity', 'seasonality_factor'
        ]].values
        
        y = features['adr'].values
        
        return X, y
        
    def _update_elasticity_model(self, bookings: pd.DataFrame):
        """Update price elasticity model."""
        # Calculate price changes
        bookings['price_change'] = bookings.groupby('arrival_date')['adr'].pct_change()
        
        # Calculate demand changes
        bookings['demand_change'] = bookings.groupby('arrival_date')['is_canceled'].pct_change()
        
        # Prepare features
        features = bookings[[
            'price_change', 'lead_time', 'market_segment',
            'room_type', 'total_of_special_requests'
        ]].dropna().values
        
        targets = bookings['demand_change'].dropna().values
        
        # Update model
        self.elasticity_model.fit(features, targets)
        
    def _update_market_model(self, market_data: pd.DataFrame):
        """Update market condition model."""
        features = market_data[[
            'demand_level', 'competition_intensity',
            'seasonality_factor', 'occupancy_rate',
            'price_sensitivity'
        ]].values
        
        # Prepare target variables (future market conditions)
        targets = np.roll(features, -1, axis=0)[:-1]
        features = features[:-1]
        
        # Update model
        self.market_model.fit(
            features,
            targets,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
    def generate_pricing_report(self) -> Dict[str, any]:
        """Generate comprehensive pricing analysis report."""
        if not self.price_history:
            return {"error": "No pricing history available"}
            
        report = {
            'current_prices': {
                'average_price': np.mean([p.final_price for p in self.price_history[-30:]]),
                'price_range': {
                    'min': min(p.final_price for p in self.price_history[-30:]),
                    'max': max(p.final_price for p in self.price_history[-30:])
                }
            },
            'price_adjustments': self._analyze_price_adjustments(),
            'demand_analysis': self._analyze_demand_patterns(),
            'elasticity_analysis': self._analyze_price_elasticity(),
            'competitive_analysis': self._analyze_competitive_position(),
            'performance_metrics': self._calculate_performance_metrics()
        }
        
        return report
        
    def _analyze_price_adjustments(self) -> Dict[str, float]:
        """Analyze historical price adjustments."""
        if not self.price_history:
            return {}
            
        adjustments = pd.DataFrame([p.adjustments for p in self.price_history])
        
        return {
            'average_adjustments': adjustments.mean().to_dict(),
            'adjustment_volatility': adjustments.std().to_dict(),
            'largest_adjustments': adjustments.abs().max().to_dict()
        }
        
    def _analyze_demand_patterns(self) -> Dict[str, float]:
        """Analyze historical demand patterns."""
        if not self.price_history:
            return {}
            
        demands = [p.expected_demand for p in self.price_history]
        
        return {
            'average_demand': float(np.mean(demands)),
            'demand_volatility': float(np.std(demands)),
            'demand_trend': float(np.polyfit(range(len(demands)), demands, 1)[0])
        }
        
    def _analyze_price_elasticity(self) -> Dict[str, float]:
        """Analyze price elasticity patterns."""
        if len(self.price_history) < 2:
            return {}
            
        elasticities = [p.price_elasticity for p in self.price_history]
        
        return {
            'average_elasticity': float(np.mean(elasticities)),
            'elasticity_volatility': float(np.std(elasticities)),
            'elasticity_trend': float(np.polyfit(range(len(elasticities)),
                                               elasticities, 1)[0])
        }
        
    def _analyze_competitive_position(self) -> Dict[str, float]:
        """Analyze competitive position history."""
        if not self.price_history:
            return {}
            
        positions = [p.competitive_position for p in self.price_history]
        
        return {
            'average_position': float(np.mean(positions)),
            'position_volatility': float(np.std(positions)),
            'position_trend': float(np.polyfit(range(len(positions)),
                                            positions, 1)[0])
        }
        
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate pricing performance metrics."""
        if not self.price_history:
            return {}
            
        return {
            'average_confidence': np.mean([p.confidence_score 
                                         for p in self.price_history]),
            'price_stability': 1 - np.std([p.final_price 
                                         for p in self.price_history]) / \
                                 np.mean([p.final_price 
                                        for p in self.price_history]),
            'adjustment_efficiency': self._calculate_adjustment_efficiency(),
            'demand_accuracy': self._calculate_demand_prediction_accuracy()
        }
        
    def _calculate_adjustment_efficiency(self) -> float:
        """Calculate efficiency of price adjustments."""
        if len(self.price_history) < 2:
            return 0.0
            
        # Calculate ratio of beneficial adjustments
        beneficial_adjustments = sum(
            1 for i in range(1, len(self.price_history))
            if self.price_history[i].expected_demand > \
               self.price_history[i-1].expected_demand
        )
        
        return beneficial_adjustments / (len(self.price_history) - 1)
        
    def _calculate_demand_prediction_accuracy(self) -> float:
        """Calculate accuracy of demand predictions."""
        if not hasattr(self, 'actual_demands'):
            return 0.0
            
        predictions = [p.expected_demand for p in self.price_history]
        actuals = self.actual_demands[:len(predictions)]
        
        return 1 - np.mean(np.abs(np.array(predictions) - np.array(actuals)))