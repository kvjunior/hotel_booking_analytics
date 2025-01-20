"""
Hotel Booking Analytics Package
-----------------------------
A comprehensive package for hotel booking analysis, prediction, and optimization.
"""

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

from .config import CONFIG
from .data_processor import DataProcessor
from .market_analyzer import MarketAnalyzer
from .network_analyzer import NetworkAnalyzer
from .risk_manager import RiskManager
from .price_optimizer import PriceOptimizer
from .booking_predictor import BookingPredictor
from .utils import setup_logger

# Setup package-wide logger
logger = setup_logger()