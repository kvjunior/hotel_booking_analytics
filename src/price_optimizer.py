"""
Price optimization system for hotel bookings.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import CONFIG
from .utils import setup_logger
from .market_analyzer import MarketRegime

logger = setup_logger(__name__)

@dataclass
class PriceRecommendation:
    """Class for storing price optimization results."""
    optimal_price: float
    price_bounds: Tuple[float, float]
    expected_revenue: float
    confidence_score: float
    supporting_factors: Dict[str, float]

class PriceOptimizer:
    """Optimizes pricing strategies for hotel bookings."""
    
    def __init__(self):
        self.demand_model = RandomForestRegressor(
            n_estimators=CONFIG['PREDICTION']['RF_N_ESTIMATORS'],
            max_depth=CONFIG['PREDICTION']['RF_MAX_DEPTH'],
            random_state=CONFIG['RANDOM_STATE']
        )
        self.revenue_history: Dict[datetime, float] = {}
        self.price_elasticity: Optional[float] = None
        
    def optimize_price(
        self,
        booking_features: pd.Series,
        market_regime: MarketRegime,
        current_occupancy: float
    ) -> PriceRecommendation:
        """Optimize price for given booking features and market conditions."""
        try:
            # Calculate base price
            base_price = self._calculate_base_price(booking_features)
            
            # Adjust bounds based on market conditions
            price_bounds = self._calculate_price_bounds(
                base_price,
                market_regime,
                current_occupancy
            )
            
            # Optimize price within bounds
            optimal_price = self._optimize_revenue(
                booking_features,
                base_price,
                price_bounds,
                market_regime
            )
            
            # Calculate expected revenue and confidence
            expected_revenue = self._calculate_expected_revenue(
                booking_features,
                optimal_price
            )
            confidence_score = self._calculate_confidence_score(
                optimal_price,
                market_regime
            )
            
            # Get supporting factors
            supporting_factors = self._get_supporting_factors(
                optimal_price,
                base_price,
                market_regime
            )
            
            recommendation = PriceRecommendation(
                optimal_price=optimal_price,
                price_bounds=price_bounds,
                expected_revenue=expected_revenue,
                confidence_score=confidence_score,
                supporting_factors=supporting_factors
            )
            
            logger.info(f"Generated price recommendation: {optimal_price:.2f}")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error in price optimization: {str(e)}")
            raise
            
    def _calculate_base_price(self, booking_features: pd.Series) -> float:
        """Calculate base price using historical data and booking features."""
        # Extract relevant features
        features = {
            'room_type': booking_features['reserved_room_type'],
            'total_guests': booking_features['total_guests'],
            'stay_duration': (booking_features['stays_in_weekend_nights'] +
                            booking_features['stays_in_week_nights']),
            'lead_time': booking_features['lead_time']
        }
        
        # Apply base pricing rules
        base_price = 100  # Default base price
        
        # Adjust for room type
        room_type_multipliers = {
            'A': 1.5, 'B': 1.3, 'C': 1.1, 'D': 1.0, 'E': 0.9, 'F': 0.8, 'G': 0.7
        }
        base_price *= room_type_multipliers.get(features['room_type'], 1.0)
        
        # Adjust for number of guests
        base_price *= (1 + 0.1 * (features['total_guests'] - 1))
        
        # Adjust for stay duration
        if features['stay_duration'] > 7:
            base_price *= 0.9  # Long stay discount
        elif features['stay_duration'] < 2:
            base_price *= 1.1  # Short stay premium
            
        # Adjust for lead time
        if features['lead_time'] > 90:
            base_price *= 0.95  # Early booking discount
        elif features['lead_time'] < 7:
            base_price *= 1.15  # Last minute premium
            
        return base_price
        
    def _calculate_price_bounds(
        self,
        base_price: float,
        market_regime: MarketRegime,
        current_occupancy: float
    ) -> Tuple[float, float]:
        """Calculate price bounds based on market conditions."""
        # Base bounds
        min_multiplier = CONFIG['PRICING']['MIN_PRICE_MULTIPLIER']
        max_multiplier = CONFIG['PRICING']['MAX_PRICE_MULTIPLIER']
        
        # Adjust for market regime
        if market_regime.demand_level > 0.8:  # High demand
            min_multiplier *= 1.2
            max_multiplier *= 1.4
        elif market_regime.demand_level < 0.2:  # Low demand
            min_multiplier *= 0.8
            max_multiplier *= 0.9
            
        # Adjust for occupancy
        if current_occupancy > 0.9:
            min_multiplier *= 1.3
            max_multiplier *= 1.5
        elif current_occupancy < 0.3:
            min_multiplier *= 0.7
            max_multiplier *= 0.8
            
        # Adjust for price volatility
        if market_regime.price_volatility > 0.5:
            min_multiplier *= 0.9
            max_multiplier *= 1.1
            
        return (base_price * min_multiplier, base_price * max_multiplier)
        
    def _optimize_revenue(
        self,
        booking_features: pd.Series,
        base_price: float,
        price_bounds: Tuple[float, float],
        market_regime: MarketRegime
    ) -> float:
        """Optimize revenue using price elasticity and market conditions."""
        def revenue_objective(price):
            # Predict demand at given price
            predicted_demand = self._predict_demand(booking_features, price)
            
            # Calculate expected revenue
            revenue = price * predicted_demand
            
            # Apply market regime adjustments
            revenue *= (1 + market_regime.demand_level)
            
            # Return negative revenue for minimization
            return -revenue
            
        # Optimize price within bounds
        result = minimize(
            revenue_objective,
            x0=base_price,
            bounds=[price_bounds],
            method='L-BFGS-B'
        )
        
        return result.x[0]
        
    def _predict_demand(
        self,
        booking_features: pd.Series,
        price: float
    ) -> float:
        """Predict demand for given features and price."""
        features = booking_features.copy()
        features['adr'] = price
        
        # Transform features for prediction
        X = pd.DataFrame([features])
        
        # Predict demand
        predicted_demand = self.demand_model.predict(X)[0]
        
        # Apply sigmoid to bound between 0 and 1
        predicted_demand = 1 / (1 + np.exp(-predicted_demand))
        
        return predicted_demand
        
    def _calculate_expected_revenue(
        self,
        booking_features: pd.Series,
        price: float
    ) -> float:
        """Calculate expected revenue for given price."""
        predicted_demand = self._predict_demand(booking_features, price)
        return price * predicted_demand
        
    def _calculate_confidence_score(
        self,
        optimal_price: float,
        market_regime: MarketRegime
    ) -> float:
        """Calculate confidence score for price recommendation."""
        # Base confidence
        confidence = 0.8
        
        # Adjust for market volatility
        confidence *= (1 - market_regime.price_volatility)
        
        # Adjust for market regime certainty
        if market_regime.demand_level > 0.8 or market_regime.demand_level < 0.2:
            confidence *= 0.9  # Less confident in extreme conditions
            
        # Bound between 0 and 1
        return max(min(confidence, 1.0), 0.0)
        
    def _get_supporting_factors(
        self,
        optimal_price: float,
        base_price: float,
        market_regime: MarketRegime
    ) -> Dict[str, float]:
        """Get factors supporting the price recommendation."""
        return {
            'price_change': (optimal_price - base_price) / base_price,
            'market_demand': market_regime.demand_level,
            'price_volatility': market_regime.price_volatility,
            'seasonality': market_regime.seasonality_score,
            'competition': market_regime.competition_index
        }
        
    def update_price_elasticity(
        self,
        price_changes: pd.Series,
        demand_changes: pd.Series
    ):
        """Update price elasticity based on historical data."""
        try:
            # Calculate price elasticity
            self.price_elasticity = (
                demand_changes.pct_change() / price_changes.pct_change()
            ).mean()
            
            logger.info(f"Updated price elasticity: {self.price_elasticity:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating price elasticity: {str(e)}")
            
    def record_revenue(self, date: datetime, revenue: float):
        """Record actual revenue for a given date."""
        self.revenue_history[date] = revenue
        
    def get_optimization_report(self) -> Dict[str, any]:
        """Generate optimization performance report."""
        if not self.revenue_history:
            return {"error": "No revenue history available"}
            
        report = {
            'total_revenue': sum(self.revenue_history.values()),
            'average_revenue': np.mean(list(self.revenue_history.values())),
            'revenue_trend': self._calculate_revenue_trend(),
            'price_elasticity': self.price_elasticity
        }
        
        return report
        
    def _calculate_revenue_trend(self) -> float:
        """Calculate revenue trend over time."""
        if len(self.revenue_history) < 2:
            return 0.0
            
        revenues = pd.Series(self.revenue_history)
        return revenues.pct_change().mean()