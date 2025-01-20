# Adaptive Hotel Analytics

This repository contains the code and data for the research paper "Adaptive Hotel Analytics: A Multi-Objective Optimization Model". The paper presents a novel framework that integrates network analysis, risk assessment, and dynamic pricing for optimizing hotel management operations.

## Repository Structure

The repository is organized into the following directories:

- `data/`: Contains the dataset used in the research, including hotel booking records and engineered features.
  - `datainfos.py`: Script for handling dataset information and characteristics.
  - `hotel_booking.csv`: The main dataset file containing hotel booking records.

- `evaluation/`: Includes scripts for model evaluation and performance tracking.
  - `model_evaluator.py`: Script for evaluating the performance of individual models and the ensemble.
  - `performance_tracker.py`: Script for tracking and visualizing model performance over time.
  - `visualization.py`: Script for creating visualizations of optimization results and business impact.

- `experiments/`: Contains code for running experiments and comparisons.
  - `advanced_experiment.py`: Script for running advanced experiments with the proposed framework.
  - `baseline_experiment.py`: Script for running experiments with baseline models.
  - `comparison_experiment.py`: Script for comparing the performance of different optimization approaches.

- `models/`: Includes the implementation of various models and optimization components.
  - `adaptive_optimizer.py`: Implementation of the adaptive optimization framework.
  - `adaptive_pricing.py`: Script for dynamic pricing optimization.
  - `multi_objective.py`: Implementation of the multi-objective optimization algorithm.
  - `network_effects.py`: Script for analyzing network effects and customer relationships.
  - `risk_aware_manager.py`: Implementation of the risk assessment and mitigation framework.

- `src/`: Contains utility scripts and helper functions.
  - `_init_.py`: Initialization script for the package.
  - `booking_predictor.py`: Script for predicting hotel bookings.
  - `config.py`: Configuration file for the project.
  - `data_processor.py`: Script for data preprocessing and feature engineering.
  - `market_analyzer.py`: Script for analyzing market conditions and trends.
  - `network_analyzer.py`: Script for analyzing customer networks and relationships.
  - `price_optimizer.py`: Script for price optimization.
  - `risk_manager.py`: Script for risk assessment and management.
  - `utils.py`: Utility functions used throughout the project.
- `main.py`: The main script to run the adaptive hotel analytics framework.
- `requirements.txt`: File listing the required Python dependencies for the project.

## Getting Started

To run the code and reproduce the results from the paper, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/kvjunior/hotel_booking_analytics.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```
   python main.py
   ```

4. Explore the results and visualizations generated in the respective directories.
