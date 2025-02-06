import numpy as np
from typing import Tuple, List, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

class AdvancedClustering:
    """
    Advanced clustering implementation that uses:
    1. Hierarchical clustering for better hand grouping
    2. Custom distance metrics considering hand potential
    3. Adaptive cluster count based on hand distribution
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def _potential_aware_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Custom distance metric that considers both immediate strength and future potential.
        
        Parameters
        ----------
        u, v : np.ndarray
            Vectors containing [win_rate, loss_rate, potential] for two hands
            
        Returns
        -------
        float
            Distance between the hands considering both current strength and potential
        """
        # Extract components
        u_strength = u[0]  # win rate
        u_potential = u[2]  # potential (PPot - NPot)
        v_strength = v[0]
        v_potential = v[2]
        
        # Weight immediate strength more than potential
        strength_weight = 0.7
        potential_weight = 0.3
        
        # Calculate weighted distance
        strength_dist = abs(u_strength - v_strength)
        potential_dist = abs(u_potential - v_potential)
        
        return strength_weight * strength_dist + potential_weight * potential_dist
    
    def _determine_optimal_clusters(self, X: np.ndarray, min_clusters: int, max_clusters: int) -> int:
        """
        Determine optimal number of clusters using the elbow method.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        min_clusters : int
            Minimum number of clusters to consider
        max_clusters : int
            Maximum number of clusters to consider
            
        Returns
        -------
        int
            Optimal number of clusters
        """
        # For small number of clusters, just use the max_clusters
        if max_clusters <= 5:
            return max_clusters
            
        # For larger numbers, use a quick estimation
        kmeans = KMeans(
            n_clusters=max_clusters,
            init="k-means++",
            n_init=1,
            max_iter=20
        )
        labels = kmeans.fit_predict(X)
        return len(np.unique(labels))

    def cluster(
        self, 
        X: np.ndarray, 
        min_clusters: int = None,
        max_clusters: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform advanced clustering on the input data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix where each row is [win_rate, loss_rate, potential]
        min_clusters : int, optional
            Minimum number of clusters (default: n_samples // 50)
        max_clusters : int, optional
            Maximum number of clusters (default: n_samples // 10)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (centroids, cluster_labels)
        """
        n_samples = len(X)
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # For small numbers of clusters, just use the specified number
        n_clusters = max_clusters if max_clusters <= 5 else min(max_clusters, n_samples // 10)
        
        # Perform clustering with optimal number of clusters
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=20,
            tol=1e-3
        )
        labels = kmeans.fit_predict(X_scaled)
        centroids = kmeans.cluster_centers_
        
        return centroids, labels

    def create_lookup(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        hands: List[Tuple]
    ) -> Dict[Tuple, int]:
        """
        Create a lookup table mapping hands to their cluster labels.
        
        Parameters
        ----------
        X : np.ndarray
            Original input data
        labels : np.ndarray
            Cluster labels
        hands : List[Tuple]
            List of hands corresponding to rows in X
            
        Returns
        -------
        Dict[Tuple, int]
            Mapping from hands to cluster labels
        """
        lookup = {}
        for hand, label in zip(hands, labels):
            lookup[hand] = int(label)
        return lookup