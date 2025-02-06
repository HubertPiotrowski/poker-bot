import logging
import time
import os
from pathlib import Path
from typing import Any, Dict, List
import concurrent.futures
import multiprocessing
import gc

import joblib
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
from tqdm import tqdm

def process_river_ehs_worker(public, cards, n_simulations_river):
    """
    Global worker function to compute expected hand strength for a given river combo.
    This function is defined at module-level so it can be pickled and used with ProcessPoolExecutor.
    """
    from poker_ai.clustering.game_utility import GameUtility
    from poker_ai.clustering.enhanced_evaluator import EnhancedEvaluator
    evaluator = EnhancedEvaluator(cards)
    our_hand = public[:2]
    board = public[2:7]
    game = GameUtility(our_hand=our_hand, board=board, cards=cards)
    ehs_squared = evaluator.calculate_ehs_squared(
        our_hand=game.our_hand,
        board=game.board,
        n_samples=n_simulations_river
    )
    ppot, npot = evaluator.calculate_potential(
        our_hand=game.our_hand,
        board=game.board,
        n_samples=n_simulations_river
    )
    ehs = np.zeros(3)
    ehs[0] = ehs_squared
    ehs[1] = 1 - ehs_squared
    ehs[2] = ppot - npot
    return ehs

from poker_ai.clustering.card_combos import CardCombos
from poker_ai.clustering.game_utility import GameUtility
from poker_ai.clustering.preflop import compute_preflop_lossless_abstraction
from poker_ai.clustering.enhanced_evaluator import EnhancedEvaluator
from poker_ai.clustering.advanced_clustering import AdvancedClustering

log = logging.getLogger("poker_ai.clustering.runner")


class CardInfoLutBuilder(CardCombos):
    """
    Stores info buckets for each street when called

    Attributes
    ----------
    card_info_lut : Dict[str, Any]
        Lookup table of card combinations per betting round to a cluster id.
    centroids : Dict[str, Any]
        Centroids per betting round for use in clustering previous rounds by
        earth movers distance.
    """

    def __init__(
        self,
        n_simulations_river: int,
        n_simulations_turn: int,
        n_simulations_flop: int,
        low_card_rank: int,
        high_card_rank: int,
        save_dir: str,
    ):
        self.n_simulations_river = n_simulations_river
        self.n_simulations_turn = n_simulations_turn
        self.n_simulations_flop = n_simulations_flop
        super().__init__(
            low_card_rank, high_card_rank,
        )
        self.card_info_lut_path: Path = Path(save_dir) / "card_info_lut.joblib"
        self.centroid_path: Path = Path(save_dir) / "centroids.joblib"
        try:
            self.card_info_lut: Dict[str, Any] = joblib.load(self.card_info_lut_path)
            self.centroids: Dict[str, Any] = joblib.load(self.centroid_path)
        except FileNotFoundError:
            self.centroids: Dict[str, Any] = {}
            self.card_info_lut: Dict[str, Any] = {}
        self.enhanced_evaluator = EnhancedEvaluator(self._cards)
        self.advanced_clustering = AdvancedClustering()

    def compute(
        self, n_river_clusters: int, n_turn_clusters: int, n_flop_clusters: int,
    ):
        """Compute all clusters and save to card_info_lut dictionary.

        Will attempt to load previous progress and will save after each cluster
        is computed.
        """
        log.info("Starting computation of clusters.")
        start = time.time()
        if "pre_flop" not in self.card_info_lut:
            self.card_info_lut["pre_flop"] = compute_preflop_lossless_abstraction(
                builder=self
            )
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            gc.collect()  # Force garbage collection after dump
            
        if "river" not in self.card_info_lut:
            self.card_info_lut["river"] = self._compute_river_clusters(
                n_river_clusters,
            )
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
            gc.collect()  # Force garbage collection after dump
            
        if "turn" not in self.card_info_lut:
            self.card_info_lut["turn"] = self._compute_turn_clusters(n_turn_clusters)
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
            gc.collect()  # Force garbage collection after dump
            
        if "flop" not in self.card_info_lut:
            self.card_info_lut["flop"] = self._compute_flop_clusters(n_flop_clusters)
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
            gc.collect()  # Force garbage collection after dump
            
        end = time.time()
        log.info(f"Finished computation of clusters - took {end - start} seconds.")

    def _compute_river_clusters(self, n_river_clusters: int):
        """Compute river clusters and create lookup table."""
        log.info("Starting computation of river clusters.")
        start = time.time()
        max_workers = int(os.environ.get('MAX_WORKERS', os.cpu_count()))
        chunk_size = max(1, min(100, len(self.river) // (max_workers * 4)))  # Limit chunk size
        log.info(f"River combos count: {len(self.river)}, chunk_size: {chunk_size}, max_workers: {max_workers}")
        
        from functools import partial
        ctx = multiprocessing.get_context("spawn")
        worker_func = partial(process_river_ehs_worker, cards=self._cards, n_simulations_river=self.n_simulations_river)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            self._river_ehs = []
            effective_batch_size = chunk_size * max_workers
            total_batches = (len(self.river) + effective_batch_size - 1) // effective_batch_size
            for i in tqdm(range(0, len(self.river), effective_batch_size), desc="Processing river batches", total=total_batches):
                batch = self.river[i:i + effective_batch_size]
                batch_results = list(executor.map(worker_func, batch, chunksize=chunk_size))
                self._river_ehs.extend(batch_results)
                gc.collect()  # Force garbage collection after each batch
                
        log.info("Computing river clusters...")
        centroids, labels = self.advanced_clustering.cluster(
            self._river_ehs,
            min_clusters=max(2, n_river_clusters - 5),
            max_clusters=n_river_clusters + 5
        )
        self.centroids["river"] = centroids
        self._river_clusters = labels
        
        end = time.time()
        log.info(
            f"Finished computation of river clusters - took {end - start} seconds."
        )
        return self.create_card_lookup(self._river_clusters, self.river)

    def _compute_turn_clusters(self, n_turn_clusters: int):
        """Compute turn clusters and create lookup table."""
        log.info("Starting computation of turn clusters.")
        start = time.time()
        max_workers = int(os.environ.get('MAX_WORKERS', os.cpu_count()))
        chunk_size = max(1, min(100, len(self.turn) // (max_workers * 4)))  # Limit chunk size
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            self._turn_ehs_distributions = []
            # Process in smaller batches to manage memory
            for i in range(0, len(self.turn), chunk_size * max_workers):
                log.info(f"Processing turn batch from {i} to {i + chunk_size * max_workers}")
                batch = self.turn[i:i + chunk_size * max_workers]
                batch_results = list(executor.map(self.process_turn_ehs_distributions, batch, chunksize=chunk_size))
                self._turn_ehs_distributions.extend(batch_results)
                log.info(f"Completed turn batch from {i} to {i + chunk_size * max_workers}")
                gc.collect()  # Force garbage collection after each batch
                
        log.info("Computing turn clusters...")
        centroids, labels = self.advanced_clustering.cluster(
            self._turn_ehs_distributions,
            min_clusters=max(2, n_turn_clusters - 5),
            max_clusters=n_turn_clusters + 5
        )
        self.centroids["turn"] = centroids
        self._turn_clusters = labels
        
        end = time.time()
        log.info(f"Finished computation of turn clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._turn_clusters, self.turn)

    def _compute_flop_clusters(self, n_flop_clusters: int):
        """Compute flop clusters and create lookup table."""
        log.info("Starting computation of flop clusters.")
        start = time.time()
        max_workers = int(os.environ.get('MAX_WORKERS', os.cpu_count()))
        chunk_size = max(1, min(100, len(self.flop) // (max_workers * 4)))  # Limit chunk size
        
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            self._flop_potential_aware_distributions = []
            # Process in smaller batches to manage memory
            for i in range(0, len(self.flop), chunk_size * max_workers):
                batch = self.flop[i:i + chunk_size * max_workers]
            batch_results = list(executor.map(self.process_flop_potential_aware_distributions, batch, chunksize=chunk_size))
            self._flop_potential_aware_distributions.extend(batch_results)
            gc.collect()  # Force garbage collection after each batch
                
        log.info("Computing flop clusters...")
        centroids, labels = self.advanced_clustering.cluster(
            self._flop_potential_aware_distributions,
            min_clusters=max(2, n_flop_clusters - 5),
            max_clusters=n_flop_clusters + 5
        )
        self.centroids["flop"] = centroids
        self._flop_clusters = labels
        
        end = time.time()
        log.info(f"Finished computation of flop clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._flop_clusters, self.flop)

    def simulate_get_ehs(self, game: GameUtility,) -> np.ndarray:
        """
        Get expected hand strength object.

        Parameters
        ----------
        game : GameUtility
            GameState for help with determining winner and sampling opponent hand

        Returns
        -------
        ehs : np.ndarray
            [win_rate, loss_rate, potential]
        """
        ehs: np.ndarray = np.zeros(3)
        
        # Calculate E[HS²] and potential
        ehs_squared = self.enhanced_evaluator.calculate_ehs_squared(
            our_hand=game.our_hand,
            board=game.board,
            n_samples=self.n_simulations_river
        )
        ppot, npot = self.enhanced_evaluator.calculate_potential(
            our_hand=game.our_hand,
            board=game.board,
            n_samples=self.n_simulations_river
        )
        ehs[0] = ehs_squared
        ehs[1] = 1 - ehs_squared
        ehs[2] = ppot - npot  # Store potential in the third component
        return ehs

    def simulate_get_turn_ehs_distributions(
        self,
        available_cards: np.ndarray,
        the_board: np.ndarray,
        our_hand: np.ndarray,
    ) -> np.ndarray:
        """
        Get histogram of frequencies that a given turn situation resulted in a
        certain cluster id after a river simulation.

        Parameters
        ----------
        available_cards : np.ndarray
            Array of available cards on the turn
        the_board : np.nearray
            The board as of the turn
        our_hand : np.ndarray
            Cards our hand (Card)

        Returns
        -------
        turn_ehs_distribution : np.ndarray
            Array of counts for each cluster the turn fell into by the river
            after simulations
        """
        turn_ehs_distribution = np.zeros(len(self.centroids["river"]))
        # sample river cards and run a simulation
        for _ in range(self.n_simulations_turn):
            river_card = np.random.choice(available_cards, 1, replace=False)
            board = np.append(the_board, river_card)
            game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
            ehs = self.simulate_get_ehs(game)
            # get EMD for expected hand strength against each river centroid
            # to which does it belong?
            for idx, river_centroid in enumerate(self.centroids["river"]):
                emd = wasserstein_distance(ehs, river_centroid)
                if idx == 0:
                    min_idx = idx
                    min_emd = emd
                else:
                    if emd < min_emd:
                        min_idx = idx
                        min_emd = emd
            # now increment the cluster to which it belongs -
            turn_ehs_distribution[min_idx] += 1 / self.n_simulations_turn
        return turn_ehs_distribution

    def process_river_ehs(self, public: np.ndarray) -> np.ndarray:
        """
        Get the expected hand strength for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Expected hand strength
        """
        our_hand = public[:2]
        board = public[2:7]
        # Get expected hand strength
        game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
        return self.simulate_get_ehs(game)

    @staticmethod
    def get_available_cards(
        cards: np.ndarray, unavailable_cards: np.ndarray
    ) -> np.ndarray:
        """
        Get all cards that are available.

        Parameters
        ----------
        cards : np.ndarray
        unavailable_cards : np.array
            Cards that are not available.

        Returns
        -------
            Available cards
        """
        # Turn into set for O(1) lookup speed.
        unavailable_cards = set(unavailable_cards.tolist())
        return np.array([c for c in cards if c not in unavailable_cards])

    def process_turn_ehs_distributions(self, public: np.ndarray) -> np.ndarray:
        """
        Get the potential aware turn distribution for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Potential aware turn distributions
        """
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        # sample river cards and run a simulation
        turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
            available_cards, the_board=public[2:6], our_hand=public[:2],
        )
        return turn_ehs_distribution

    def process_flop_potential_aware_distributions(
        self, public: np.ndarray,
    ) -> np.ndarray:
        """
        Get the potential aware flop distribution for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Potential aware flop distributions
        """
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        potential_aware_distribution_flop = np.zeros(len(self.centroids["turn"]))
        for j in range(self.n_simulations_flop):
            # randomly generating turn
            turn_card = np.random.choice(available_cards, 1, replace=False)
            our_hand = public[:2]
            board = public[2:5]
            the_board = np.append(board, turn_card).tolist()
            # getting available cards
            available_cards_turn = np.array(
                [x for x in available_cards if x != turn_card[0]]
            )
            turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
                available_cards_turn, the_board=the_board, our_hand=our_hand,
            )
            for idx, turn_centroid in enumerate(self.centroids["turn"]):
                # earth mover distance
                emd = wasserstein_distance(turn_ehs_distribution, turn_centroid)
                if idx == 0:
                    min_idx = idx
                    min_emd = emd
                else:
                    if emd < min_emd:
                        min_idx = idx
                        min_emd = emd
            # Now increment the cluster to which it belongs.
            potential_aware_distribution_flop[min_idx] += 1 / self.n_simulations_flop
        return potential_aware_distribution_flop

    @staticmethod
    def create_card_lookup(clusters: np.ndarray, card_combos: np.ndarray) -> Dict:
        """
        Create lookup table.

        Parameters
        ----------
        clusters : np.ndarray
            Array of cluster ids.
        card_combos : np.ndarray
            The card combos to which the cluster ids belong.

        Returns
        -------
        lossy_lookup : Dict
            Lookup table for finding cluster ids.
        """
        log.info("Creating lookup table.")
        lossy_lookup = {}
        for i, card_combo in enumerate(tqdm(card_combos)):
            lossy_lookup[tuple(card_combo)] = clusters[i]
        return lossy_lookup
