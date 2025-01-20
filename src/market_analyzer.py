"""
Market regime detection and analysis for hotel booking patterns.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass

from .config import CONFIG
from .utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class MarketRegime:
    """Class for storing market regime information."""
    name: str
    demand_level: float
    price_volatility: float
    seasonality_score: float
    competition_index: float

class MarketAnalyzer:
    """Analyzes market conditions and detects regimes."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.regime_classifier = KMeans(
            n_clusters=4,  # High/Low demand x High/Low volatility
            random_state=CONFIG['RANDOM_STATE']
        )
        self.current_regime: Optional[MarketRegime] = None
        
    def analyze_market_conditions(
        self,
        df: pd.DataFrame,
        window_size: int = 30
    ) -> MarketRegime:
        """Analyze current market conditions."""
        try:
            demand_level = self._calculate_demand_level(df)
            price_volatility = self._calculate_price_volatility(df, window_size)
            seasonality_score = self._calculate_seasonality_score(df)
            competition_index = self._calculate_competition_index(df)
            
            regime = self._classify_regime(
                demand_level,
                price_volatility,
                seasonality_score,
                competition_index
            )
            
            self.current_regime = regime
            logger.info(f"Current market regime: {regime.name}")
            
            return regime
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            raise
            
    def _calculate_demand_level(self, df: pd.DataFrame) -> float:
        """Calculate current demand level."""
        recent_bookings = df.sort_values('arrival_date').tail(
            CONFIG['MARKET_REGIMES']['VOLATILITY_WINDOW']
        )
        
        # Calculate occupancy rate
        occupancy_rate = (
            recent_bookings['total_guests'].sum() /
            (len(recent_bookings) * recent_bookings['adults'].max())
        )
        
        # Calculate booking pace
        booking_pace = recent_bookings['lead_time'].mean()
        
        # Combine metrics
        demand_level = 0.7 * occupancy_rate + 0.3 * (1 - booking_pace/365)
        
        return demand_level
        
    def _calculate_price_volatility(
        self,
        df: pd.DataFrame,
        window_size: int
    ) -> float:
        """Calculate price volatility."""
        recent_prices = df.sort_values('arrival_date').tail(window_size)['adr']
        return recent_prices.std() / recent_prices.mean()
        
    def _calculate_seasonality_score(self, df: pd.DataFrame) -> float:
        """Calculate seasonality score."""
        # Group by month and calculate average demand
        monthly_demand = df.groupby('arrival_date_month')['total_guests'].mean()
        
        # Calculate coefficient of variation
        seasonality_score = monthly_demand.std() / monthly_demand.mean()
        
        return seasonality_score
        
    def _calculate_competition_index(self, df: pd.DataFrame) -> float:
        """Calculate competition index based on price sensitivity."""
        # Calculate price elasticity
        avg_price = df['adr'].mean()
        avg_demand = df['total_guests'].mean()
        
        price_high = df[df['adr'] > avg_price]['total_guests'].mean()
        price_low = df[df['adr'] <= avg_price]['total_guests'].mean()
        
        price_elasticity = (price_high - price_low) / (price_high + price_low)
        
        return abs(price_elasticity)
        
    def _classify_regime(
        self,
        demand_level: float,
        price_volatility: float,
        seasonality_score: float,
        competition_index: float
    ) -> MarketRegime:
        """Classify market regime based on indicators."""
        # Create feature vector
        features = np.array([
            demand_level,
            price_volatility,
            seasonality_score,
            competition_index
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Classify regime
        regime_id = self.regime_classifier.fit_predict(features_scaled)[0]
        
        # Map regime ID to meaningful name
        regime_names = {
            0: 'High Demand, Low Volatility',
            1: 'High Demand, High Volatility',
            2: 'Low Demand, Low Volatility',
            3: 'Low Demand, High Volatility'
        }
        
        return MarketRegime(
            name=regime_names[regime_id],
            demand_level=demand_level,
            price_volatility=price_volatility,
            seasonality_score=seasonality_score,
            competition_index=competition_index
        )
        
    def get_market_recommendations(self) -> Dict[str, str]:
        """Get recommendations based on current market regime."""
        if not self.current_regime:
            return {"error": "No market regime analyzed yet"}
            
        recommendations = {
            'High Demand, Low Volatility': {
                'pricing': 'Implement premium pricing strategies',
                'marketing': 'Focus on high-value customer segments',
                'inventory': 'Maintain strict booking controls'
            },
            'High Demand, High Volatility': {
                'pricing': 'Use dynamic pricing with price fences',
                'marketing': 'Target last-minute bookers',
                'inventory': 'Keep flexible room allocation'
            },
            'Low Demand, Low Volatility': {
                'pricing': 'Maintain stable, competitive rates',
                'marketing': 'Focus on advance purchase discounts',
                'inventory': 'Consider room upgrades'
            },
            'Low Demand, High Volatility': {
                'pricing': 'Implement aggressive promotions',
                'marketing': 'Target multiple segments',
                'inventory': 'Maximize occupancy'
            }
        }
        
        return recommendations.get(self.current_regime.name, {})