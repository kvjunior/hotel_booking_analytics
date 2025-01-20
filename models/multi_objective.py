"""
Multi-objective optimization system for hotel management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta

from ..config import CONFIG
from ..utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class ObjectiveWeight:
    """Weight configuration for different objectives."""
    revenue: float = 0.4
    occupancy: float = 0.3
    satisfaction: float = 0.2
    efficiency: float = 0.1

@dataclass
class OptimizationConstraints:
    """Constraints for optimization."""
    min_price: float
    max_price: float
    min_occupancy: float
    max_occupancy: float
    min_satisfaction: float
    max_staff_hours: float
    min_service_level: float

@dataclass
class OptimizationResult:
    """Results from multi-objective optimization."""
    optimal_params: Dict[str, float]
    objective_values: Dict[str, float]
    pareto_front: List[Dict[str, float]]
    convergence_history: List[float]
    constraint_satisfaction: Dict[str, bool]

class MultiObjectiveOptimizer:
    """Multi-objective optimization system for hotel operations."""
    
    def __init__(
        self,
        weights: Optional[ObjectiveWeight] = None,
        scaler: Optional[MinMaxScaler] = None
    ):
        self.weights = weights or ObjectiveWeight()
        self.scaler = scaler or MinMaxScaler()
        self.optimization_history: List[OptimizationResult] = []
        self.pareto_front: List[Dict[str, float]] = []
        
    def optimize(
        self,
        initial_params: Dict[str, float],
        constraints: OptimizationConstraints,
        historical_data: pd.DataFrame
    ) -> OptimizationResult:
        """Perform multi-objective optimization."""
        try:
            logger.info("Starting multi-objective optimization")
            
            # Define bounds for optimization variables
            bounds = self._create_bounds(constraints)
            
            # Define objective functions
            objectives = {
                'revenue': self._revenue_objective(historical_data),
                'occupancy': self._occupancy_objective(historical_data),
                'satisfaction': self._satisfaction_objective(historical_data),
                'efficiency': self._efficiency_objective(historical_data)
            }
            
            # Create constraint functions
            constraint_funcs = self._create_constraints(constraints)
            
            # Initialize optimization
            x0 = self._prepare_initial_params(initial_params)
            
            # Perform optimization
            result = minimize(
                self._combined_objective(objectives),
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_funcs,
                options={'maxiter': 1000}
            )
            
            # Process results
            optimal_params = self._process_optimization_results(result)
            
            # Calculate objective values
            objective_values = self._calculate_objective_values(
                optimal_params,
                objectives
            )
            
            # Update Pareto front
            self._update_pareto_front(optimal_params, objective_values)
            
            # Create result object
            optimization_result = OptimizationResult(
                optimal_params=optimal_params,
                objective_values=objective_values,
                pareto_front=self.pareto_front,
                convergence_history=result.fun_history \
                    if hasattr(result, 'fun_history') else [],
                constraint_satisfaction=self._check_constraints(
                    optimal_params,
                    constraints
                )
            )
            
            # Store result in history
            self.optimization_history.append(optimization_result)
            
            logger.info("Multi-objective optimization completed successfully")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            raise
            
    def _revenue_objective(
        self,
        historical_data: pd.DataFrame
    ) -> Callable[[np.ndarray], float]:
        """Create revenue objective function."""
        def revenue_func(x: np.ndarray) -> float:
            price, occupancy = x[0], x[1]
            return -(price * occupancy * 100)  # Negative for minimization
        return revenue_func
        
    def _occupancy_objective(
        self,
        historical_data: pd.DataFrame
    ) -> Callable[[np.ndarray], float]:
        """Create occupancy objective function."""
        def occupancy_func(x: np.ndarray) -> float:
            price, occupancy = x[0], x[1]
            demand_elasticity = -0.5  # Example elasticity coefficient
            expected_occupancy = occupancy * \
                np.exp(demand_elasticity * (price / 100 - 1))
            return -expected_occupancy  # Negative for minimization
        return occupancy_func
        
    def _satisfaction_objective(
        self,
        historical_data: pd.DataFrame
    ) -> Callable[[np.ndarray], float]:
        """Create customer satisfaction objective function."""
        def satisfaction_func(x: np.ndarray) -> float:
            price, occupancy, service_level = x[0], x[1], x[2]
            # Simple satisfaction model based on service level and occupancy
            satisfaction = service_level * (1 - occupancy/2) * \
                         (1 + np.log(price/100))
            return -satisfaction  # Negative for minimization
        return satisfaction_func
        
    def _efficiency_objective(
        self,
        historical_data: pd.DataFrame
    ) -> Callable[[np.ndarray], float]:
        """Create operational efficiency objective function."""
        def efficiency_func(x: np.ndarray) -> float:
            occupancy, service_level, staff_hours = x[1], x[2], x[3]
            # Efficiency metric based on staff utilization
            efficiency = staff_hours / (occupancy * service_level * 100)
            return efficiency  # Already positive for minimization
        return efficiency_func
        
    def _combined_objective(
        self,
        objectives: Dict[str, Callable]
    ) -> Callable[[np.ndarray], float]:
        """Create combined objective function with weights."""
        def combined_func(x: np.ndarray) -> float:
            weighted_sum = (
                self.weights.revenue * objectives['revenue'](x) +
                self.weights.occupancy * objectives['occupancy'](x) +
                self.weights.satisfaction * objectives['satisfaction'](x) +
                self.weights.efficiency * objectives['efficiency'](x)
            )
            return weighted_sum
        return combined_func
        
    def _create_bounds(
        self,
        constraints: OptimizationConstraints
    ) -> List[Tuple[float, float]]:
        """Create bounds for optimization variables."""
        return [
            (constraints.min_price, constraints.max_price),  # Price
            (constraints.min_occupancy, constraints.max_occupancy),  # Occupancy
            (constraints.min_service_level, 1.0),  # Service level
            (0, constraints.max_staff_hours)  # Staff hours
        ]
        
    def _create_constraints(
        self,
        constraints: OptimizationConstraints
    ) -> List[Dict]:
        """Create constraint functions for optimization."""
        return [
            {'type': 'ineq', 'fun': lambda x: constraints.max_price - x[0]},
            {'type': 'ineq', 'fun': lambda x: x[0] - constraints.min_price},
            {'type': 'ineq', 'fun': lambda x: x[2] - constraints.min_service_level},
            {'type': 'ineq', 'fun': lambda x: constraints.max_staff_hours - x[3]}
        ]
        
    def _prepare_initial_params(
        self,
        initial_params: Dict[str, float]
    ) -> np.ndarray:
        """Prepare initial parameters for optimization."""
        return np.array([
            initial_params.get('price', 100),
            initial_params.get('occupancy', 0.7),
            initial_params.get('service_level', 0.8),
            initial_params.get('staff_hours', 160)
        ])
        
    def _process_optimization_results(
        self,
        result: any
    ) -> Dict[str, float]:
        """Process optimization results into parameter dictionary."""
        return {
            'price': float(result.x[0]),
            'occupancy': float(result.x[1]),
            'service_level': float(result.x[2]),
            'staff_hours': float(result.x[3])
        }
        
    def _calculate_objective_values(
        self,
        params: Dict[str, float],
        objectives: Dict[str, Callable]
    ) -> Dict[str, float]:
        """Calculate individual objective values for optimal parameters."""
        x = np.array([
            params['price'],
            params['occupancy'],
            params['service_level'],
            params['staff_hours']
        ])
        
        return {
            name: -float(func(x))  # Convert back to maximization
            for name, func in objectives.items()
        }
        
    def _update_pareto_front(
        self,
        params: Dict[str, float],
        objective_values: Dict[str, float]
    ):
        """Update Pareto front with new solution."""
        solution = {**params, **objective_values}
        
        # Check if solution is dominated by any existing solution
        is_dominated = any(
            all(existing[obj] >= solution[obj] for obj in objective_values.keys())
            and any(existing[obj] > solution[obj] for obj in objective_values.keys())
            for existing in self.pareto_front
        )
        
        if not is_dominated:
            # Remove solutions that this one dominates
            self.pareto_front = [
                existing for existing in self.pareto_front
                if not all(solution[obj] >= existing[obj] 
                          for obj in objective_values.keys())
            ]
            self.pareto_front.append(solution)
            
    def _check_constraints(
        self,
        params: Dict[str, float],
        constraints: OptimizationConstraints
    ) -> Dict[str, bool]:
        """Check if solution satisfies all constraints."""
        return {
            'price_constraints': constraints.min_price <= params['price'] <= \
                               constraints.max_price,
            'occupancy_constraints': constraints.min_occupancy <= \
                                   params['occupancy'] <= \
                                   constraints.max_occupancy,
            'service_constraints': params['service_level'] >= \
                                 constraints.min_service_level,
            'staff_constraints': params['staff_hours'] <= \
                               constraints.max_staff_hours
        }
        
    def plot_pareto_front(self) -> None:
        """Plot the Pareto front for visualization."""
        if not self.pareto_front:
            logger.warning("No Pareto front available for plotting")
            return
            
        # Extract objective values
        revenue = [sol['revenue'] for sol in self.pareto_front]
        occupancy = [sol['occupancy'] for sol in self.pareto_front]
        satisfaction = [sol['satisfaction'] for sol in self.pareto_front]
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=revenue,
            y=occupancy,
            z=satisfaction,
            mode='markers',
            marker=dict(
                size=8,
                color=satisfaction,
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title='Pareto Front Visualization',
            scene=dict(
                xaxis_title='Revenue',
                yaxis_title='Occupancy',
                zaxis_title='Satisfaction'
            )
        )
        
        # Save plot
        fig.write_html("pareto_front.html")
        logger.info("Pareto front plot saved as 'pareto_front.html'")