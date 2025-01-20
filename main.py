"""
Main execution script for hotel booking analytics system.
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

from src.data_processor import DataProcessor
from src.market_analyzer import MarketAnalyzer
from src.network_analyzer import NetworkAnalyzer
from src.risk_manager import RiskManager
from src.price_optimizer import PriceOptimizer
from src.booking_predictor import BookingPredictor
from src.utils import setup_logger

from models.adaptive_optimizer import AdaptiveHotelOptimizer
from models.multi_objective import MultiObjectiveOptimizer
from models.risk_aware_manager import RiskAwareManager
from models.network_effects import BookingNetworkAnalyzer
from models.adaptive_pricing import AdaptivePricingSystem

from experiments.baseline_experiment import BaselineExperiment
from experiments.advanced_experiment import AdvancedExperiment
from experiments.comparison_experiment import ModelComparison

from evaluation.model_evaluator import ModelEvaluator
from evaluation.performance_tracker import PerformanceTracker
from evaluation.visualization import VisualizationManager

logger = setup_logger(__name__)

class HotelAnalyticsSystem:
    """Main class for running the hotel booking analytics system."""
    
    def __init__(self, config_path: str = None):
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.market_analyzer = MarketAnalyzer()
        self.network_analyzer = NetworkAnalyzer()
        self.risk_manager = RiskManager()
        self.price_optimizer = PriceOptimizer()
        self.booking_predictor = BookingPredictor()
        
        # Initialize advanced models
        self.adaptive_optimizer = AdaptiveHotelOptimizer()
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        self.risk_aware_manager = RiskAwareManager()
        self.network_effects_analyzer = BookingNetworkAnalyzer()
        self.adaptive_pricing_system = AdaptivePricingSystem()
        
        # Initialize evaluation components
        self.model_evaluator = ModelEvaluator()
        self.performance_tracker = PerformanceTracker()
        self.viz_manager = VisualizationManager()
        
    def run_analysis(self, data_path: str):
        """Run complete analysis pipeline."""
        try:
            logger.info("Starting analysis pipeline")
            
            # Load and preprocess data
            df = self.data_processor.load_data(data_path)
            df = self.data_processor.preprocess_data(df)
            
            # Market analysis
            market_regime = self.market_analyzer.analyze_market_conditions(df)
            
            # Network analysis
            booking_network = self.network_analyzer.build_booking_network(df)
            network_metrics = self.network_analyzer.analyze_network()
            
            # Risk assessment
            df = self.risk_manager.detect_anomalies(df)
            risk_profiles = []
            for _, booking in df.iterrows():
                risk_assessment = self.risk_manager.assess_booking_risk(booking)
                risk_profiles.append(risk_assessment)
                
            # Price optimization
            price_recommendations = []
            for _, booking in df.iterrows():
                recommendation = self.price_optimizer.optimize_price(
                    booking,
                    market_regime,
                    network_metrics.connectivity_metrics['density']
                )
                price_recommendations.append(recommendation)
                
            # Booking prediction
            booking_predictions = self.booking_predictor.predict(df)
            
            # Save analysis results
            self._save_analysis_results(
                market_regime,
                network_metrics,
                risk_profiles,
                price_recommendations,
                booking_predictions
            )
            
            logger.info("Analysis pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            raise
            
    def run_experiments(self):
        """Run model experiments and comparisons."""
        try:
            logger.info("Starting experiments")
            
            # Create experiment output directory
            experiment_dir = self.output_dir / 'experiments'
            experiment_dir.mkdir(exist_ok=True)
            
            # Run baseline experiments
            baseline_experiment = BaselineExperiment()
            baseline_experiment.run_experiments(
                'data/hotel_booking.csv',
                str(experiment_dir)
            )
            
            # Run advanced experiments
            advanced_experiment = AdvancedExperiment()
            advanced_experiment.run_experiments(
                'data/hotel_booking.csv',
                str(experiment_dir)
            )
            
            # Run model comparison
            model_comparison = ModelComparison()
            model_comparison.compare_models(
                str(experiment_dir / 'baseline_results.json'),
                str(experiment_dir / 'advanced_results.json'),
                str(experiment_dir)
            )
            
            logger.info("Experiments completed successfully")
            
        except Exception as e:
            logger.error(f"Error running experiments: {str(e)}")
            raise
            
    def _save_analysis_results(
        self,
        market_regime: any,
        network_metrics: any,
        risk_profiles: list,
        price_recommendations: list,
        booking_predictions: list
    ):
        """Save analysis results to output directory."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = self.output_dir / f'analysis_{timestamp}'
            results_dir.mkdir(exist_ok=True)
            
            # Save market analysis
            with open(results_dir / 'market_analysis.json', 'w') as f:
                json.dump(market_regime.__dict__, f, indent=4)
                
            # Save network analysis
            with open(results_dir / 'network_analysis.json', 'w') as f:
                json.dump(network_metrics.__dict__, f, indent=4)
                
            # Save risk profiles
            with open(results_dir / 'risk_profiles.json', 'w') as f:
                json.dump([profile.__dict__ for profile in risk_profiles], f, indent=4)
                
            # Save price recommendations
            with open(results_dir / 'price_recommendations.json', 'w') as f:
                json.dump([rec.__dict__ for rec in price_recommendations], f, indent=4)
                
            # Save booking predictions
            with open(results_dir / 'booking_predictions.json', 'w') as f:
                json.dump(booking_predictions, f, indent=4)
                
            logger.info(f"Analysis results saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
            raise

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Hotel Booking Analytics System')
    parser.add_argument(
        '--data',
        type=str,
        default='data/hotel_booking.csv',
        help='Path to hotel booking dataset'
    )
    parser.add_argument(
        '--run-experiments',
        action='store_true',
        help='Run model experiments'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = HotelAnalyticsSystem()
        
        # Run analysis
        system.run_analysis(args.data)
        
        # Run experiments if requested
        if args.run_experiments:
            system.run_experiments()
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()