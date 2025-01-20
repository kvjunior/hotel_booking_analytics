"""
Risk management system for hotel bookings.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import CONFIG
from .utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class RiskAssessment:
    """Class for storing risk assessment results."""
    risk_score: float
    risk_factors: Dict[str, float]
    risk_category: str
    recommended_actions: List[str]

class RiskManager:
    """Manages booking-related risks."""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=CONFIG['RANDOM_STATE']
        )
        self.scaler = StandardScaler()
        self.risk_history: Dict[int, RiskAssessment] = {}
        
    def assess_booking_risk(self, booking: pd.Series) -> RiskAssessment:
        """Assess the risk level of a booking."""
        try:
            # Calculate individual risk components
            cancellation_risk = self._calculate_cancellation_risk(booking)
            payment_risk = self._calculate_payment_risk(booking)
            occupancy_risk = self._calculate_occupancy_risk(booking)
            seasonality_risk = self._calculate_seasonality_risk(booking)
            
            # Combine risk components
            risk_factors = {
                'cancellation_risk': cancellation_risk,
                'payment_risk': payment_risk,
                'occupancy_risk': occupancy_risk,
                'seasonality_risk': seasonality_risk
            }
            
            # Calculate overall risk score
            risk_score = self._calculate_overall_risk(risk_factors)
            
            # Determine risk category and recommendations
            risk_category = self._categorize_risk(risk_score)
            recommended_actions = self._get_risk_recommendations(
                risk_category,
                risk_factors
            )
            
            # Create risk assessment
            assessment = RiskAssessment(
                risk_score=risk_score,
                risk_factors=risk_factors,
                risk_category=risk_category,
                recommended_actions=recommended_actions
            )
            
            # Store assessment in history
            self.risk_history[booking.name] = assessment
            
            logger.info(f"Completed risk assessment for booking {booking.name}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            raise
            
    def _calculate_cancellation_risk(self, booking: pd.Series) -> float:
        """Calculate risk of booking cancellation."""
        # Factors that influence cancellation risk
        risk_factors = {
            'lead_time': booking['lead_time'] / 365,  # Normalize to [0,1]
            'previous_cancellations': min(booking['previous_cancellations'] / 3, 1),
            'deposit_type': 1 if booking['deposit_type'] == 'No Deposit' else 0,
            'booking_changes': min(booking['booking_changes'] / 3, 1)
        }
        
        # Weighted sum of risk factors
        weights = {
            'lead_time': 0.3,
            'previous_cancellations': 0.3,
            'deposit_type': 0.2,
            'booking_changes': 0.2
        }
        
        return sum(factor * weights[name] 
                  for name, factor in risk_factors.items())
                  
    def _calculate_payment_risk(self, booking: pd.Series) -> float:
        """Calculate payment-related risks."""
        # Payment risk factors
        risk_factors = {
            'adr_ratio': booking['adr'] / booking['total_guests'] 
                        if booking['total_guests'] > 0 else 1,
            'deposit': 1 if booking['deposit_type'] == 'No Deposit' else 0,
            'market_segment': 0.5 if booking['market_segment'] == 'Groups' else 0
        }
        
        # Normalize adr_ratio to [0,1] using sigmoid
        risk_factors['adr_ratio'] = 1 / (1 + np.exp(-risk_factors['adr_ratio']))
        
        weights = {
            'adr_ratio': 0.4,
            'deposit': 0.4,
            'market_segment': 0.2
        }
        
        return sum(factor * weights[name] 
                  for name, factor in risk_factors.items())
                  
    def _calculate_occupancy_risk(self, booking: pd.Series) -> float:
        """Calculate occupancy-related risks."""
        # Occupancy risk factors
        risk_factors = {
            'stay_duration': min(
                (booking['stays_in_weekend_nights'] + 
                 booking['stays_in_week_nights']) / 14,
                1
            ),
            'special_requests': min(
                booking['total_of_special_requests'] / 5,
                1
            ),
            'room_assigned': 1 if booking['assigned_room_type'] != \
                              booking['reserved_room_type'] else 0
        }
        
        weights = {
            'stay_duration': 0.4,
            'special_requests': 0.3,
            'room_assigned': 0.3
        }
        
        return sum(factor * weights[name] 
                  for name, factor in risk_factors.items())
                  
    def _calculate_seasonality_risk(self, booking: pd.Series) -> float:
        """Calculate seasonality-related risks."""
        # Define high-risk periods (example)
        high_risk_months = ['July', 'August', 'December']
        high_risk_days = [5, 6]  # Weekend days
        
        risk_factors = {
            'high_season': 1 if booking['arrival_date_month'] in high_risk_months else 0,
            'weekend': 1 if booking['arrival_date_day_of_month'] % 7 in high_risk_days else 0,
            'lead_time_season': booking['lead_time'] / 365
        }
        
        weights = {
            'high_season': 0.4,
            'weekend': 0.3,
            'lead_time_season': 0.3
        }
        
        return sum(factor * weights[name] 
                  for name, factor in risk_factors.items())
                  
    def _calculate_overall_risk(self, risk_factors: Dict[str, float]) -> float:
        """Calculate overall risk score."""
        weights = {
            'cancellation_risk': 0.35,
            'payment_risk': 0.25,
            'occupancy_risk': 0.20,
            'seasonality_risk': 0.20
        }
        
        return sum(risk_factors[name] * weights[name] 
                  for name in risk_factors.keys())
                  
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level."""
        if risk_score >= CONFIG['RISK_MANAGEMENT']['RISK_THRESHOLD']:
            return 'High Risk'
        elif risk_score >= CONFIG['RISK_MANAGEMENT']['RISK_THRESHOLD'] * 0.7:
            return 'Medium Risk'
        else:
            return 'Low Risk'
            
    def _get_risk_recommendations(
        self,
        risk_category: str,
        risk_factors: Dict[str, float]
    ) -> List[str]:
        """Get recommendations based on risk assessment."""
        recommendations = []
        
        if risk_category == 'High Risk':
            if risk_factors['cancellation_risk'] > 0.7:
                recommendations.append('Require full prepayment')
                recommendations.append('Implement strict cancellation policy')
                
            if risk_factors['payment_risk'] > 0.7:
                recommendations.append('Request deposit or payment guarantee')
                recommendations.append('Verify payment method in advance')
                
            if risk_factors['occupancy_risk'] > 0.7:
                recommendations.append('Consider overbooking protection')
                recommendations.append('Monitor room allocation closely')
                
        elif risk_category == 'Medium Risk':
            if risk_factors['cancellation_risk'] > 0.5:
                recommendations.append('Request partial prepayment')
                
            if risk_factors['payment_risk'] > 0.5:
                recommendations.append('Verify customer details')
                
            if risk_factors['occupancy_risk'] > 0.5:
                recommendations.append('Prepare alternative room options')
                
        else:  # Low Risk
            recommendations.append('Standard booking procedures apply')
            
        return recommendations
        
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalous bookings."""
        try:
            # Prepare features for anomaly detection
            features = df[['lead_time', 'adr', 'total_guests', 
                         'booking_changes', 'previous_cancellations']]
            features_scaled = self.scaler.fit_transform(features)
            
            # Detect anomalies
            anomaly_labels = self.anomaly_detector.fit_predict(features_scaled)
            
            # Add anomaly labels to dataframe
            df['is_anomaly'] = anomaly_labels == -1
            
            n_anomalies = df['is_anomaly'].sum()
            logger.info(f"Detected {n_anomalies} anomalous bookings")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise
            
    def generate_risk_report(self) -> Dict[str, any]:
        """Generate summary report of risk assessments."""
        if not self.risk_history:
            return {"error": "No risk assessments available"}
            
        report = {
            'total_assessments': len(self.risk_history),
            'risk_distribution': {
                'High Risk': 0,
                'Medium Risk': 0,
                'Low Risk': 0
            },
            'average_risk_score': 0,
            'risk_factors_summary': {
                'cancellation_risk': 0,
                'payment_risk': 0,
                'occupancy_risk': 0,
                'seasonality_risk': 0
            }
        }
        
        # Calculate statistics
        for assessment in self.risk_history.values():
            report['risk_distribution'][assessment.risk_category] += 1
            report['average_risk_score'] += assessment.risk_score
            
            for factor, value in assessment.risk_factors.items():
                report['risk_factors_summary'][factor] += value
                
        # Calculate averages
        report['average_risk_score'] /= len(self.risk_history)
        for factor in report['risk_factors_summary']:
            report['risk_factors_summary'][factor] /= len(self.risk_history)
            
        return report