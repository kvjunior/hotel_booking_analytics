"""
Network effects analysis system for hotel booking patterns and customer relationships.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import networkx as nx
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import community
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats

from ..config import CONFIG
from ..utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class NetworkMetrics:
    """Container for network analysis metrics."""
    centrality_scores: Dict[str, float]
    community_detection: Dict[str, int]
    influence_scores: Dict[str, float]
    connectivity_metrics: Dict[str, float]
    temporal_patterns: Dict[str, any]

@dataclass
class CustomerSegment:
    """Customer segment information."""
    segment_id: int
    size: int
    avg_value: float
    characteristics: Dict[str, any]
    connection_strength: float
    influence_score: float

class BookingNetworkAnalyzer:
    """Analyzes booking patterns using network theory and graph analytics."""
    
    def __init__(self):
        self.G = nx.Graph()
        self.customer_graph = nx.Graph()
        self.temporal_graph = nx.Graph()
        self.scaler = StandardScaler()
        self.metrics_history: List[NetworkMetrics] = []
        
    def build_network(
        self,
        booking_data: pd.DataFrame,
        customer_data: Optional[pd.DataFrame] = None
    ):
        """Construct multi-layer network representation of bookings."""
        try:
            # Reset graphs
            self.G.clear()
            self.customer_graph.clear()
            self.temporal_graph.clear()
            
            # Build main booking network
            self._build_booking_network(booking_data)
            
            # Build customer relationship network if data available
            if customer_data is not None:
                self._build_customer_network(customer_data)
                
            # Build temporal patterns network
            self._build_temporal_network(booking_data)
            
            logger.info("Successfully built multi-layer network representation")
            
        except Exception as e:
            logger.error(f"Error building network: {str(e)}")
            raise
            
    def analyze_network(self) -> NetworkMetrics:
        """Perform comprehensive network analysis."""
        try:
            # Calculate centrality measures
            centrality_scores = self._calculate_centrality_metrics()
            
            # Detect communities
            community_structure = self._detect_communities()
            
            # Calculate influence scores
            influence_scores = self._calculate_influence_scores()
            
            # Calculate connectivity metrics
            connectivity_metrics = self._calculate_connectivity_metrics()
            
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns()
            
            # Create metrics object
            metrics = NetworkMetrics(
                centrality_scores=centrality_scores,
                community_detection=community_structure,
                influence_scores=influence_scores,
                connectivity_metrics=connectivity_metrics,
                temporal_patterns=temporal_patterns
            )
            
            # Store metrics history
            self.metrics_history.append(metrics)
            
            logger.info("Network analysis completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in network analysis: {str(e)}")
            raise
            
    def identify_customer_segments(
        self,
        booking_data: pd.DataFrame
    ) -> List[CustomerSegment]:
        """Identify and analyze customer segments using network structure."""
        try:
            segments = []
            
            # Detect communities in customer network
            communities = community.best_partition(
                self.customer_graph,
                resolution=1.2
            )
            
            # Analyze each community
            for community_id in set(communities.values()):
                # Get nodes (customers) in this community
                community_nodes = [
                    node for node, comm in communities.items()
                    if comm == community_id
                ]
                
                # Calculate segment characteristics
                segment = self._analyze_segment(
                    community_nodes,
                    booking_data
                )
                
                segments.append(segment)
                
            logger.info(f"Identified {len(segments)} customer segments")
            return segments
            
        except Exception as e:
            logger.error(f"Error in customer segmentation: {str(e)}")
            raise
            
    def find_influential_customers(
        self,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """Identify most influential customers based on network metrics."""
        try:
            # Calculate customer influence scores
            influence_scores = {}
            
            for node in self.customer_graph.nodes():
                # Combine multiple centrality measures
                degree_cent = nx.degree_centrality(self.customer_graph)[node]
                betweenness_cent = nx.betweenness_centrality(self.customer_graph)[node]
                eigenvector_cent = nx.eigenvector_centrality(self.customer_graph)[node]
                
                # Calculate total influence score
                influence_scores[node] = (
                    0.4 * degree_cent +
                    0.3 * betweenness_cent +
                    0.3 * eigenvector_cent
                )
                
            # Sort and get top N influential customers
            influential_customers = sorted(
                influence_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            logger.info(f"Identified {top_n} most influential customers")
            return influential_customers
            
        except Exception as e:
            logger.error(f"Error finding influential customers: {str(e)}")
            raise
            
    def _build_booking_network(self, booking_data: pd.DataFrame):
        """Build network representation of booking patterns."""
        for idx, booking in booking_data.iterrows():
            # Add booking node
            self.G.add_node(
                idx,
                booking_date=booking['arrival_date'],
                room_type=booking['reserved_room_type'],
                price=booking['adr']
            )
            
            # Connect similar bookings
            for other_idx, other_booking in booking_data.iterrows():
                if idx != other_idx:
                    similarity = self._calculate_booking_similarity(
                        booking,
                        other_booking
                    )
                    if similarity > CONFIG['NETWORKS']['MIN_EDGE_WEIGHT']:
                        self.G.add_edge(idx, other_idx, weight=similarity)
                        
    def _build_customer_network(self, customer_data: pd.DataFrame):
        """Build network representation of customer relationships."""
        for idx, customer in customer_data.iterrows():
            self.customer_graph.add_node(
                customer['customer_id'],
                value=customer['total_spend'],
                frequency=customer['booking_frequency']
            )
            
        # Connect customers based on similarities
        customer_features = customer_data[
            ['total_spend', 'booking_frequency', 'avg_stay_length']
        ].values
        
        similarities = cosine_similarity(customer_features)
        
        for i in range(len(customer_data)):
            for j in range(i + 1, len(customer_data)):
                if similarities[i, j] > CONFIG['NETWORKS']['MIN_EDGE_WEIGHT']:
                    self.customer_graph.add_edge(
                        customer_data.iloc[i]['customer_id'],
                        customer_data.iloc[j]['customer_id'],
                        weight=similarities[i, j]
                    )
                    
    def _build_temporal_network(self, booking_data: pd.DataFrame):
        """Build network representation of temporal booking patterns."""
        # Group bookings by date
        daily_bookings = booking_data.groupby('arrival_date').size()
        
        # Create nodes for each date
        for date, count in daily_bookings.items():
            self.temporal_graph.add_node(
                date,
                booking_count=count
            )
            
        # Connect consecutive dates
        dates = sorted(daily_bookings.index)
        for i in range(len(dates) - 1):
            self.temporal_graph.add_edge(
                dates[i],
                dates[i + 1],
                weight=abs(daily_bookings[dates[i]] - daily_bookings[dates[i + 1]])
            )
            
    def _calculate_booking_similarity(
        self,
        booking1: pd.Series,
        booking2: pd.Series
    ) -> float:
        """Calculate similarity between two bookings."""
        # Define feature weights
        weights = {
            'adr': 0.3,
            'room_type': 0.2,
            'stay_length': 0.2,
            'booking_changes': 0.1,
            'special_requests': 0.2
        }
        
        similarity = 0.0
        
        # Price similarity
        price_diff = abs(booking1['adr'] - booking2['adr'])
        similarity += weights['adr'] * (1 / (1 + price_diff))
        
        # Room type similarity
        if booking1['reserved_room_type'] == booking2['reserved_room_type']:
            similarity += weights['room_type']
            
        # Stay length similarity
        stay1 = booking1['stays_in_weekend_nights'] + booking1['stays_in_week_nights']
        stay2 = booking2['stays_in_weekend_nights'] + booking2['stays_in_week_nights']
        stay_diff = abs(stay1 - stay2)
        similarity += weights['stay_length'] * (1 / (1 + stay_diff))
        
        # Booking changes similarity
        if booking1['booking_changes'] == booking2['booking_changes']:
            similarity += weights['booking_changes']
            
        # Special requests similarity
        if booking1['total_of_special_requests'] == \
           booking2['total_of_special_requests']:
            similarity += weights['special_requests']
            
        return similarity
        
    def _calculate_centrality_metrics(self) -> Dict[str, float]:
        """Calculate various centrality metrics."""
        metrics = {}
        
        # Degree centrality
        metrics.update({
            f"degree_centrality_{node}": value
            for node, value in nx.degree_centrality(self.G).items()
        })
        
        # Betweenness centrality
        metrics.update({
            f"betweenness_centrality_{node}": value
            for node, value in nx.betweenness_centrality(self.G).items()
        })
        
        # Eigenvector centrality
        metrics.update({
            f"eigenvector_centrality_{node}": value
            for node, value in nx.eigenvector_centrality(self.G).items()
        })
        
        return metrics
        
    def _detect_communities(self) -> Dict[str, int]:
        """Detect communities in the network."""
        return community.best_partition(self.G)
        
    def _calculate_influence_scores(self) -> Dict[str, float]:
        """Calculate influence scores for nodes."""
        influence_scores = {}
        
        for node in self.G.nodes():
            # Combine various centrality measures
            degree_cent = nx.degree_centrality(self.G)[node]
            betweenness_cent = nx.betweenness_centrality(self.G)[node]
            eigenvector_cent = nx.eigenvector_centrality(self.G)[node]
            
            # Calculate weighted influence score
            influence_scores[str(node)] = (
                0.4 * degree_cent +
                0.3 * betweenness_cent +
                0.3 * eigenvector_cent
            )
            
        return influence_scores
        
    def _calculate_connectivity_metrics(self) -> Dict[str, float]:
        """Calculate network connectivity metrics."""
        return {
            'density': nx.density(self.G),
            'average_clustering': nx.average_clustering(self.G),
            'average_shortest_path': \
                nx.average_shortest_path_length(self.G) \
                if nx.is_connected(self.G) else float('inf'),
            'diameter': nx.diameter(self.G) \
                if nx.is_connected(self.G) else float('inf')
        }
        
    def _analyze_temporal_patterns(self) -> Dict[str, any]:
        """Analyze temporal patterns in the network."""
        temporal_metrics = {
            'seasonality': self._calculate_seasonality(),
            'trend': self._calculate_trend(),
            'volatility': self._calculate_volatility()
        }
        
        return temporal_metrics
        
    def _calculate_seasonality(self) -> float:
        """Calculate seasonality score."""
        if not self.temporal_graph:
            return 0.0
            
        booking_counts = [
            data['booking_count']
            for _, data in self.temporal_graph.nodes(data=True)
        ]
        
        # Use FFT to detect periodic patterns
        fft = np.fft.fft(booking_counts)
        return float(np.abs(fft[1:]).max() / len(booking_counts))
        
    def _calculate_trend(self) -> float:
        """Calculate trend in booking patterns."""
        if not self.temporal_graph:
            return 0.0
            
        booking_counts = [
            data['booking_count']
            for _, data in self.temporal_graph.nodes(data=True)
        ]
        
        # Use linear regression to detect trend
        x = np.arange(len(booking_counts))
        slope, _, _, _, _ = stats.linregress(x, booking_counts)
        return float(slope)
        
    def _calculate_volatility(self) -> float:
        """Calculate booking pattern volatility."""
        if not self.temporal_graph:
            return 0.0
            
        booking_counts = [
            data['booking_count']
            for _, data in self.temporal_graph.nodes(data=True)
        ]
        
        return float(np.std(booking_counts) / np.mean(booking_counts))
        
    def _analyze_segment(
        self,
        community_nodes: List[str],
        booking_data: pd.DataFrame
    ) -> CustomerSegment:
        """Analyze characteristics of a customer segment."""
        segment_bookings = booking_data[
            booking_data['customer_id'].isin(community_nodes)
        ]
        
        # Calculate segment metrics
        avg_value = segment_bookings['adr'].mean()
        
        # Calculate segment characteristics
        characteristics = {
            'avg_stay_length': segment_bookings['stays_in_week_nights'].mean() + \
                             segment_bookings['stays_in_weekend_nights'].mean(),
            'preferred_room_type': segment_bookings['reserved_room_type'].mode()[0],
            'booking_lead_time': segment_bookings['lead_time'].mean(),
            'special_requests_rate': segment_bookings['total_of_special_requests'].mean()
        }
        
        # Calculate connection strength within segment
        connection_strength = nx.density(
            self.customer_graph.subgraph(community_nodes)
        )
        
        # Calculate segment influence
        influence_score = np.mean([
            self._calculate_influence_scores()[node]
            for node in community_nodes
            if node in self._calculate_influence_scores()
        ])
        
        return CustomerSegment(
            segment_id=len(self.metrics_history),
            size=len(community_nodes),
            avg_value=float(avg_value),
            characteristics=characteristics,
            connection_strength=float(connection_strength),
            influence_score=float(influence_score)
        )
        
    def generate_network_report(self) -> Dict[str, any]:
        """Generate comprehensive network analysis report."""
        if not self.metrics_history:
            return {"error": "No network analysis history available"}
            
        latest_metrics = self.metrics_history[-1]
        
        report = {
            'global_metrics': {
                'network_density': latest_metrics.connectivity_metrics['density'],
                'average_clustering': latest_metrics.connectivity_metrics['average_clustering'],
                'network_diameter': latest_metrics.connectivity_metrics['diameter']
            },
            'community_structure': {
                'number_of_communities': len(set(latest_metrics.community_detection.values())),
                'modularity_score': self._calculate_modularity(latest_metrics.community_detection)
            },
            'influence_analysis': {
                'top_influential_nodes': sorted(
                    latest_metrics.influence_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            },
            'temporal_analysis': latest_metrics.temporal_patterns,
            'network_evolution': self._analyze_network_evolution()
        }
        
        return report
        
    def _calculate_modularity(self, communities: Dict[str, int]) -> float:
        """Calculate modularity score for community structure."""
        try:
            return community.modularity(communities, self.G)
        except Exception:
            return 0.0
            
    def _analyze_network_evolution(self) -> Dict[str, any]:
        """Analyze how network metrics have evolved over time."""
        if len(self.metrics_history) < 2:
            return {}
            
        # Track evolution of key metrics
        density_trend = [m.connectivity_metrics['density'] 
                        for m in self.metrics_history]
        clustering_trend = [m.connectivity_metrics['average_clustering'] 
                          for m in self.metrics_history]
        community_sizes = [len(set(m.community_detection.values())) 
                         for m in self.metrics_history]
        
        return {
            'density_evolution': {
                'trend': np.polyfit(range(len(density_trend)), density_trend, 1)[0],
                'volatility': np.std(density_trend)
            },
            'clustering_evolution': {
                'trend': np.polyfit(range(len(clustering_trend)), 
                                  clustering_trend, 1)[0],
                'volatility': np.std(clustering_trend)
            },
            'community_evolution': {
                'trend': np.polyfit(range(len(community_sizes)), 
                                  community_sizes, 1)[0],
                'volatility': np.std(community_sizes)
            }
        }