"""
Network analysis for hotel booking patterns and customer relationships.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import community
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass

from .config import CONFIG
from .utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class NetworkMetrics:
    """Class for storing network analysis metrics."""
    centrality_scores: Dict[int, float]
    community_labels: Dict[int, int]
    influence_scores: Dict[int, float]
    connectivity_metrics: Dict[str, float]

class NetworkAnalyzer:
    """Analyzes booking patterns using network analysis techniques."""
    
    def __init__(self):
        self.G = nx.Graph()
        self.metrics: Optional[NetworkMetrics] = None
        
    def build_booking_network(self, df: pd.DataFrame) -> nx.Graph:
        """Build network from booking data."""
        try:
            self.G.clear()
            
            # Add nodes (bookings)
            for idx, row in df.iterrows():
                self.G.add_node(
                    idx,
                    booking_date=row['arrival_date'],
                    total_guests=row['total_guests'],
                    adr=row['adr']
                )
            
            # Add edges based on similarities
            self._add_temporal_edges(df)
            self._add_guest_similarity_edges(df)
            self._add_booking_pattern_edges(df)
            
            logger.info(f"Built network with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
            return self.G
            
        except Exception as e:
            logger.error(f"Error building network: {str(e)}")
            raise
            
    def _add_temporal_edges(self, df: pd.DataFrame):
        """Add edges between temporally close bookings."""
        dates = pd.to_datetime(df['arrival_date'])
        
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                time_diff = abs((dates.iloc[i] - dates.iloc[j]).days)
                if time_diff <= 7:  # Connect bookings within a week
                    weight = 1 - (time_diff / 7)
                    if weight >= CONFIG['NETWORKS']['MIN_EDGE_WEIGHT']:
                        self.G.add_edge(i, j, weight=weight, type='temporal')
                        
    def _add_guest_similarity_edges(self, df: pd.DataFrame):
        """Add edges based on guest similarity."""
        guest_features = [
            'adults', 'children', 'babies',
            'meal', 'market_segment', 'customer_type'
        ]
        
        # Calculate guest similarity matrix
        guest_matrix = pd.get_dummies(df[guest_features]).values
        similarities = cosine_similarity(guest_matrix)
        
        # Add edges for similar guests
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if similarities[i, j] >= CONFIG['NETWORKS']['MIN_EDGE_WEIGHT']:
                    self.G.add_edge(i, j, weight=similarities[i, j], type='guest')
                    
    def _add_booking_pattern_edges(self, df: pd.DataFrame):
        """Add edges based on booking patterns."""
        booking_features = [
            'lead_time', 'total_nights', 'adr',
            'booking_changes', 'total_of_special_requests'
        ]
        
        # Calculate booking pattern similarity matrix
        booking_matrix = df[booking_features].values
        booking_matrix = (booking_matrix - booking_matrix.mean(axis=0)) / booking_matrix.std(axis=0)
        similarities = cosine_similarity(booking_matrix)
        
        # Add edges for similar booking patterns
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if similarities[i, j] >= CONFIG['NETWORKS']['MIN_EDGE_WEIGHT']:
                    self.G.add_edge(i, j, weight=similarities[i, j], type='pattern')
                    
    def analyze_network(self) -> NetworkMetrics:
        """Perform comprehensive network analysis."""
        try:
            # Calculate centrality measures
            centrality_scores = nx.eigenvector_centrality(self.G, weight='weight')
            
            # Detect communities
            community_labels = community.best_partition(
                self.G,
                weight='weight',
                resolution=CONFIG['NETWORKS']['COMMUNITY_RESOLUTION']
            )
            
            # Calculate influence scores
            influence_scores = self._calculate_influence_scores()
            
            # Calculate connectivity metrics
            connectivity_metrics = {
                'density': nx.density(self.G),
                'average_clustering': nx.average_clustering(self.G, weight='weight'),
                'average_shortest_path': nx.average_shortest_path_length(self.G, weight='weight')
                if nx.is_connected(self.G) else float('inf')
            }
            
            self.metrics = NetworkMetrics(
                centrality_scores=centrality_scores,
                community_labels=community_labels,
                influence_scores=influence_scores,
                connectivity_metrics=connectivity_metrics
            )
            
            logger.info("Network analysis completed successfully")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error in network analysis: {str(e)}")
            raise
            
    def _calculate_influence_scores(self) -> Dict[int, float]:
        """Calculate influence scores for each node."""
        # Combine multiple centrality measures
        degree_cent = nx.degree_centrality(self.G)
        closeness_cent = nx.closeness_centrality(self.G, distance='weight')
        betweenness_cent = nx.betweenness_centrality(self.G, weight='weight')
        
        influence_scores = {}
        for node in self.G.nodes():
            influence_scores[node] = (
                0.4 * degree_cent[node] +
                0.3 * closeness_cent[node] +
                0.3 * betweenness_cent[node]
            )
            
        return influence_scores
        
    def get_recommendations(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Get recommendations based on network analysis."""
        if not self.metrics:
            return {"error": "Network analysis not performed yet"}
            
        recommendations = {
            'high_influence_customers': [],
            'community_representatives': [],
            'potential_promoters': []
        }
        
        # Identify high influence customers
        influence_threshold = np.percentile(
            list(self.metrics.influence_scores.values()),
            90
        )
        recommendations['high_influence_customers'] = [
            node for node, score in self.metrics.influence_scores.items()
            if score >= influence_threshold
        ]
        
        # Identify community representatives
        for community_id in set(self.metrics.community_labels.values()):
            community_nodes = [
                node for node, comm in self.metrics.community_labels.items()
                if comm == community_id
            ]
            if community_nodes:
                representative = max(
                    community_nodes,
                    key=lambda x: self.metrics.centrality_scores[x]
                )
                recommendations['community_representatives'].append(representative)
                
        # Identify potential promoters
        promoter_threshold = np.percentile(
            list(self.metrics.centrality_scores.values()),
            85
        )
        recommendations['potential_promoters'] = [
            node for node, score in self.metrics.centrality_scores.items()
            if score >= promoter_threshold
        ]
        
        return recommendations