"""
Risk-aware booking management system with advanced risk assessment and mitigation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from scipy.stats import norm

from ..config import CONFIG
from ..utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class RiskProfile:
    """Comprehensive risk profile for a booking."""
    cancellation_risk: float
    payment_risk: float
    occupancy_risk: float
    revenue_risk: float
    overall_risk_score: float
    risk_factors: Dict[str, float]
    confidence_score: float

@dataclass
class MitigationStrategy:
    """Risk mitigation strategy details."""
    recommended_actions: List[str]
    expected_impact: Dict[str, float]
    implementation_priority: str
    cost_benefit_ratio: float

class RiskAwareManager:
    """Advanced risk management system for hotel bookings."""
    
    def __init__(self):
        self._initialize_models()
        self.scaler = StandardScaler()
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        self.risk_history: List[RiskProfile] = []
        
    def _initialize_models(self):
        """Initialize risk assessment models."""
        # Anomaly detection model
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=CONFIG['RANDOM_STATE']
        )
        
        # Cancellation prediction model
        self.cancellation_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(None, 10)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.cancellation_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Revenue risk model
        self.revenue_model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        self.revenue_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
    def assess_booking_risk(
        self,
        booking_data: pd.Series,
        historical_context: Optional[pd.DataFrame] = None
    ) -> RiskProfile:
        """Perform comprehensive risk assessment for a booking."""
        try:
            # Calculate individual risk components
            cancellation_risk = self._assess_cancellation_risk(
                booking_data,
                historical_context
            )
            
            payment_risk = self._assess_payment_risk(booking_data)
            
            occupancy_risk = self._assess_occupancy_risk(
                booking_data,
                historical_context
            )
            
            revenue_risk = self._assess_revenue_risk(
                booking_data,
                historical_context
            )
            
            # Calculate risk factors
            risk_factors = self._calculate_risk_factors(
                booking_data,
                cancellation_risk,
                payment_risk,
                occupancy_risk,
                revenue_risk
            )
            
            # Calculate overall risk score
            overall_risk = self._calculate_overall_risk(risk_factors)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(risk_factors)
            
            # Create risk profile
            risk_profile = RiskProfile(
                cancellation_risk=cancellation_risk,
                payment_risk=payment_risk,
                occupancy_risk=occupancy_risk,
                revenue_risk=revenue_risk,
                overall_risk_score=overall_risk,
                risk_factors=risk_factors,
                confidence_score=confidence_score
            )
            
            # Update risk history
            self.risk_history.append(risk_profile)
            
            logger.info(f"Risk assessment completed with score: {overall_risk:.2f}")
            return risk_profile
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            raise
            
    def generate_mitigation_strategy(
        self,
        risk_profile: RiskProfile,
        booking_data: pd.Series
    ) -> MitigationStrategy:
        """Generate risk mitigation strategy based on risk profile."""
        try:
            recommended_actions = []
            expected_impact = {}
            
            # Handle high cancellation risk
            if risk_profile.cancellation_risk > self.risk_thresholds['high']:
                actions, impact = self._mitigate_cancellation_risk(booking_data)
                recommended_actions.extend(actions)
                expected_impact.update(impact)
                
            # Handle high payment risk
            if risk_profile.payment_risk > self.risk_thresholds['high']:
                actions, impact = self._mitigate_payment_risk(booking_data)
                recommended_actions.extend(actions)
                expected_impact.update(impact)
                
            # Handle high occupancy risk
            if risk_profile.occupancy_risk > self.risk_thresholds['high']:
                actions, impact = self._mitigate_occupancy_risk(booking_data)
                recommended_actions.extend(actions)
                expected_impact.update(impact)
                
            # Calculate implementation priority
            priority = self._calculate_priority(risk_profile, expected_impact)
            
            # Calculate cost-benefit ratio
            cost_benefit_ratio = self._calculate_cost_benefit_ratio(
                recommended_actions,
                expected_impact
            )
            
            strategy = MitigationStrategy(
                recommended_actions=recommended_actions,
                expected_impact=expected_impact,
                implementation_priority=priority,
                cost_benefit_ratio=cost_benefit_ratio
            )
            
            logger.info(f"Generated mitigation strategy with {len(recommended_actions)} actions")
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating mitigation strategy: {str(e)}")
            raise
            
    def _assess_cancellation_risk(
        self,
        booking: pd.Series,
        historical_data: Optional[pd.DataFrame]
    ) -> float:
        """Assess risk of booking cancellation."""
        # Prepare features for cancellation prediction
        features = np.array([
            booking['lead_time'],
            booking['previous_cancellations'],
            booking['booking_changes'],
            booking['adr'],
            booking['total_of_special_requests']
        ]).reshape(1, -1)
        
        features_scaled = self.scaler.fit_transform(features)
        
        # Get base cancellation probability
        cancellation_prob = float(self.cancellation_model.predict(features_scaled))
        
        # Adjust based on historical patterns
        if historical_data is not None:
            historical_rate = historical_data['is_canceled'].mean()
            cancellation_prob = 0.7 * cancellation_prob + 0.3 * historical_rate
            
        return cancellation_prob
        
    def _assess_payment_risk(self, booking: pd.Series) -> float:
        """Assess payment-related risks."""
        # Calculate base payment risk
        base_risk = 0.0
        
        # Risk factors and their weights
        risk_factors = {
            'no_deposit': booking['deposit_type'] == 'No Deposit',
            'high_adr': booking['adr'] > 300,
            'corporate_booking': booking['market_segment'] == 'Corporate',
            'first_time_guest': booking['is_repeated_guest'] == 0
        }
        
        weights = {
            'no_deposit': 0.4,
            'high_adr': 0.3,
            'corporate_booking': -0.2,  # Reduces risk
            'first_time_guest': 0.2
        }
        
        # Calculate weighted risk score
        for factor, present in risk_factors.items():
            if present:
                base_risk += weights[factor]
                
        # Normalize to [0, 1]
        return max(min(base_risk, 1.0), 0.0)
        
    def _assess_occupancy_risk(
        self,
        booking: pd.Series,
        historical_data: Optional[pd.DataFrame]
    ) -> float:
        """Assess occupancy-related risks."""
        base_risk = 0.3  # Base risk level
        
        # Risk factors
        if booking['stays_in_weekend_nights'] + \
           booking['stays_in_week_nights'] > 7:
            base_risk += 0.2  # Long stays have higher risk
            
        if booking['adults'] + booking['children'] + \
           booking['babies'] > 3:
            base_risk += 0.1  # Larger groups have higher risk
            
        if historical_data is not None:
            # Check historical occupancy patterns
            similar_periods = historical_data[
                historical_data['arrival_date_month'] == \
                booking['arrival_date_month']
            ]
            if len(similar_periods) > 0:
                historical_occupancy = similar_periods['is_canceled'].mean()
                base_risk = 0.7 * base_risk + 0.3 * historical_occupancy
                
        return min(base_risk, 1.0)
        
    def _assess_revenue_risk(
        self,
        booking: pd.Series,
        historical_data: Optional[pd.DataFrame]
    ) -> float:
        """Assess revenue-related risks."""
        # Prepare features for revenue risk assessment
        features = np.array([
            booking['adr'],
            booking['stays_in_weekend_nights'] + booking['stays_in_week_nights'],
            booking['adults'] + booking['children'],
            booking['is_repeated_guest'],
            booking['previous_bookings_not_canceled']
        ]).reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        
        # Predict revenue risk
        predicted_risk = float(self.revenue_model.predict(features_scaled))
        
        # Normalize to [0, 1]
        return norm.cdf(predicted_risk)
        
    def _calculate_risk_factors(
        self,
        booking: pd.Series,
        cancellation_risk: float,
        payment_risk: float,
        occupancy_risk: float,
        revenue_risk: float
    ) -> Dict[str, float]:
        """Calculate detailed risk factors."""
        return {
            'lead_time_risk': min(booking['lead_time'] / 365, 1.0),
            'price_risk': min(booking['adr'] / 1000, 1.0),
            'guest_history_risk': 0.8 if booking['is_repeated_guest'] == 0 else 0.2,
            'booking_change_risk': min(booking['booking_changes'] / 5, 1.0),
            'market_segment_risk': 0.6 if booking['market_segment'] == 'Groups' else 0.3,
            'cancellation_risk': cancellation_risk,
            'payment_risk': payment_risk,
            'occupancy_risk': occupancy_risk,
            'revenue_risk': revenue_risk
        }
        
    def _calculate_overall_risk(self, risk_factors: Dict[str, float]) -> float:
        """Calculate overall risk score from risk factors."""
        weights = {
            'lead_time_risk': 0.1,
            'price_risk': 0.15,
            'guest_history_risk': 0.1,
            'booking_change_risk': 0.05,
            'market_segment_risk': 0.1,
            'cancellation_risk': 0.2,
            'payment_risk': 0.15,
            'occupancy_risk': 0.1,
            'revenue_risk': 0.05
        }
        
        weighted_risk = sum(
            risk * weights[factor]
            for factor, risk in risk_factors.items()
        )
        
        return min(weighted_risk, 1.0)
        
    def _calculate_confidence_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate confidence score for risk assessment."""
        # Base confidence
        confidence = 0.8
        
        # Reduce confidence for extreme risk values
        for risk in risk_factors.values():
            if risk > 0.9 or risk < 0.1:
                confidence *= 0.9
                
        # Reduce confidence for high volatility in risk factors
        risk_std = np.std(list(risk_factors.values()))
        confidence *= (1 - risk_std)
        
        return max(min(confidence, 1.0), 0.0)
        
    def _mitigate_cancellation_risk(
        self,
        booking: pd.Series
    ) -> Tuple[List[str], Dict[str, float]]:
        """Generate cancellation risk mitigation strategies."""
        actions = []
        impact = {}
        
        if booking['deposit_type'] == 'No Deposit':
            actions.append("Require deposit payment")
            impact['reduced_cancellation_risk'] = 0.3
            
        if booking['lead_time'] > 60:
            actions.append("Offer early bird discount with non-refundable policy")
            impact['revenue_protection'] = 0.25
            
        actions.append("Implement flexible rescheduling policy")
        impact['customer_satisfaction'] = 0.2
        
        return actions, impact
        
    def _mitigate_payment_risk(
        self,
        booking: pd.Series
    ) -> Tuple[List[str], Dict[str, float]]:
        """Generate payment risk mitigation strategies."""
        actions = []
        impact = {}
        
        if booking['adr'] > 200:
            actions.append("Require staged payment plan")
            impact['payment_security'] = 0.4
            
        actions.append("Verify payment method in advance")
        impact['fraud_risk_reduction'] = 0.3
        
        return actions, impact
        
    def _mitigate_occupancy_risk(
        self,
        booking: pd.Series
    ) -> Tuple[List[str], Dict[str, float]]:
        """Generate occupancy risk mitigation strategies."""
        actions = []
        impact = {}
        
        if booking['stays_in_week_nights'] + \
           booking['stays_in_weekend_nights'] > 5:
            actions.append("Implement progressive room pricing")
            impact['revenue_optimization'] = 0.25
            
        actions.append("Prepare backup room allocation plan")
        impact['operational_flexibility'] = 0.2
        
        return actions, impact
        
    def _calculate_priority(
        self,
        risk_profile: RiskProfile,
        expected_impact: Dict[str, float]
    ) -> str:
        """Calculate implementation priority for mitigation strategy."""
        total_impact = sum(expected_impact.values())
        
        if risk_profile.overall_risk_score > 0.8 and total_impact > 0.5:
            return "High"
        elif risk_profile.overall_risk_score > 0.5 and total_impact > 0.3:
            return "Medium"
        else:
            return "Low"
            
    def _calculate_cost_benefit_ratio(
        self,
        actions: List[str],
        expected_impact: Dict[str, float]
    ) -> float:
        """Calculate cost-benefit ratio for mitigation strategy."""
        # Estimated costs for different types of actions
        action_costs = {
            'Require deposit payment': 0.1,
            'Offer early bird discount': 0.3,
            'Implement flexible rescheduling': 0.2,
            'Require staged payment': 0.15,
            'Verify payment method': 0.05,
            'Implement progressive pricing': 0.25,
            'Prepare backup allocation': 0.1
        }
        
        # Calculate total cost
        total_cost = sum(
            action_costs.get(action, 0.2)  # Default cost 0.2 for unknown actions
            for action in actions
        )
        
        # Calculate total benefit
        total_benefit = sum(expected_impact.values())
        
        # Avoid division by zero
        if total_cost == 0:
            return float('inf')
            
        return total_benefit / total_cost
        
    def update_risk_models(
        self,
        new_data: pd.DataFrame,
        actual_outcomes: pd.Series
    ):
        """Update risk assessment models with new data."""
        try:
            # Update anomaly detector
            self.anomaly_detector.fit(self.scaler.fit_transform(new_data))
            
            # Update cancellation model
            features = self._prepare_features_for_cancellation(new_data)
            self.cancellation_model.fit(
                features,
                actual_outcomes,
                epochs=5,
                batch_size=32,
                verbose=0
            )
            
            # Update revenue model
            revenue_features = self._prepare_features_for_revenue(new_data)
            actual_revenue = new_data['adr'] * \
                           (new_data['stays_in_weekend_nights'] + \
                            new_data['stays_in_week_nights'])
            self.revenue_model.fit(
                revenue_features,
                actual_revenue,
                epochs=5,
                batch_size=32,
                verbose=0
            )
            
            logger.info("Risk models updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating risk models: {str(e)}")
            raise
            
    def _prepare_features_for_cancellation(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """Prepare features for cancellation model."""
        features = data[['lead_time', 'previous_cancellations',
                        'booking_changes', 'adr',
                        'total_of_special_requests']].values
        return self.scaler.fit_transform(features)
        
    def _prepare_features_for_revenue(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """Prepare features for revenue model."""
        features = data[['adr', 'stays_in_weekend_nights',
                        'stays_in_week_nights', 'adults',
                        'is_repeated_guest']].values
        return self.scaler.fit_transform(features)
        
    def generate_risk_report(self) -> Dict[str, any]:
        """Generate comprehensive risk assessment report."""
        if not self.risk_history:
            return {"error": "No risk assessments available"}
            
        report = {
            'total_assessments': len(self.risk_history),
            'average_risk_score': np.mean([r.overall_risk_score 
                                         for r in self.risk_history]),
            'risk_distribution': {
                'high': len([r for r in self.risk_history 
                           if r.overall_risk_score > self.risk_thresholds['high']]),
                'medium': len([r for r in self.risk_history 
                             if self.risk_thresholds['medium'] < \
                             r.overall_risk_score <= self.risk_thresholds['high']]),
                'low': len([r for r in self.risk_history 
                          if r.overall_risk_score <= self.risk_thresholds['medium']])
            },
            'average_confidence': np.mean([r.confidence_score 
                                         for r in self.risk_history]),
            'risk_factors_summary': self._calculate_risk_factors_summary(),
            'trend_analysis': self._analyze_risk_trends()
        }
        
        return report
        
    def _calculate_risk_factors_summary(self) -> Dict[str, float]:
        """Calculate summary statistics for risk factors."""
        all_factors = {}
        for profile in self.risk_history:
            for factor, value in profile.risk_factors.items():
                if factor not in all_factors:
                    all_factors[factor] = []
                all_factors[factor].append(value)
                
        return {
            factor: {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values)
            }
            for factor, values in all_factors.items()
        }
        
    def _analyze_risk_trends(self) -> Dict[str, any]:
        """Analyze trends in risk assessments over time."""
        if len(self.risk_history) < 2:
            return {}
            
        risk_scores = [r.overall_risk_score for r in self.risk_history]
        trend = np.polyfit(range(len(risk_scores)), risk_scores, 1)[0]
        
        return {
            'risk_score_trend': float(trend),
            'trend_direction': 'increasing' if trend > 0.01 else
                             'decreasing' if trend < -0.01 else 'stable',
            'volatility': float(np.std(risk_scores))
        }