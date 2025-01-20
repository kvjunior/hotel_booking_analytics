"""
Configuration settings for the hotel booking analytics system.
"""

import os
from pathlib import Path

CONFIG = {
    # Paths
    'DATA_DIR': Path('../data'),
    'MODELS_DIR': Path('../models'),
    'OUTPUT_DIR': Path('../output'),
    
    # Data Processing
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.2,
    'VALIDATION_SIZE': 0.2,
    
    # Feature Engineering
    'CATEGORICAL_FEATURES': [
        'hotel', 'meal', 'market_segment', 'distribution_channel',
        'reserved_room_type', 'deposit_type', 'customer_type'
    ],
    'NUMERICAL_FEATURES': [
        'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
        'adults', 'children', 'babies', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes',
        'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
        'total_of_special_requests'
    ],
    'DATE_FEATURES': [
        'arrival_date_year', 'arrival_date_month', 
        'arrival_date_week_number', 'arrival_date_day_of_month'
    ],
    
    # Model Parameters
    'NETWORKS': {
        'MIN_EDGE_WEIGHT': 0.1,
        'COMMUNITY_RESOLUTION': 1.0
    },
    
    'RISK_MANAGEMENT': {
        'RISK_THRESHOLD': 0.7,
        'HIGH_RISK_MULTIPLIER': 1.5,
        'LOW_RISK_MULTIPLIER': 0.8
    },
    
    'PRICING': {
        'MIN_PRICE_MULTIPLIER': 0.5,
        'MAX_PRICE_MULTIPLIER': 2.0,
        'PRICE_INCREMENT': 0.01,
        'LEARNING_RATE': 0.01
    },
    
    'PREDICTION': {
        'RF_N_ESTIMATORS': 100,
        'RF_MAX_DEPTH': 10,
        'XGB_LEARNING_RATE': 0.1,
        'XGB_MAX_DEPTH': 7,
        'NN_HIDDEN_LAYERS': [64, 32, 16],
        'NN_DROPOUT_RATE': 0.2
    },
    
    # Market Analysis
    'MARKET_REGIMES': {
        'HIGH_DEMAND_THRESHOLD': 0.8,
        'LOW_DEMAND_THRESHOLD': 0.2,
        'VOLATILITY_WINDOW': 30
    },
    
    # GPU Settings
    'USE_GPU': True,
    'GPU_MEMORY_FRACTION': 0.8,
    
    # Logging
    'LOG_LEVEL': 'INFO',
    'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Environment-specific overrides
if os.environ.get('ENVIRONMENT') == 'production':
    CONFIG.update({
        'LOG_LEVEL': 'WARNING',
        'TEST_SIZE': 0.1,
        'GPU_MEMORY_FRACTION': 0.9
    })

# Create necessary directories
for dir_path in [CONFIG['DATA_DIR'], CONFIG['MODELS_DIR'], CONFIG['OUTPUT_DIR']]:
    dir_path.mkdir(parents=True, exist_ok=True)