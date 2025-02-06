import numpy as np
from typing import List, Tuple, Dict
from poker_ai.poker.evaluation import Evaluator
from poker_ai.poker.card import Card

class EnhancedEvaluator:
    """
    Enhanced hand strength evaluator that considers:
    1. Current hand strength (HS)
    2. Positive potential (PPot) - probability of improving
    3. Negative potential (NPot) - probability of falling behind
    4. E[HS²] - combination of current strength and future potential
    """

    def __init__(self, cards: np.ndarray):
        self._evaluator = Evaluator()
        self._cards = cards
        self._hand_cache: Dict[Tuple[int, ...], float] = {}

    def evaluate_hand(self, hand: np.ndarray, board: np.ndarray) -> int:
        """Evaluate a specific hand with caching for performance."""
        key = tuple(sorted(hand.tolist() + board.tolist()))
        if key not in self._hand_cache:
            self._hand_cache[key] = self._evaluator.evaluate(
                board=board.astype(np.int32).tolist(),
                cards=hand.astype(np.int32).tolist(),
            )
        return self._hand_cache[key]

    def get_available_cards(self, unavailable_cards: np.ndarray) -> np.ndarray:
        """Get all available cards for sampling."""
        unavailable_set = set(unavailable_cards.tolist())
        return np.array([c for c in self._cards if c not in unavailable_set])

    def calculate_hand_strength(
        self, 
        our_hand: np.ndarray, 
        board: np.ndarray,
        n_samples: int = 100
    ) -> float:
        """
        Calculate immediate hand strength through Monte Carlo simulation.
        
        Parameters
        ----------
        our_hand : np.ndarray
            Our hole cards
        board : np.ndarray
            Current board cards
        n_samples : int
            Number of Monte Carlo samples
            
        Returns
        -------
        float
            Hand strength [0,1] representing probability of winning
        """
        wins = 0
        available_cards = self.get_available_cards(np.concatenate([our_hand, board]))
        
        our_rank = self.evaluate_hand(our_hand, board)
        
        for _ in range(n_samples):
            # Sample opponent hand
            opp_hand = np.random.choice(available_cards, 2, replace=False)
            opp_rank = self.evaluate_hand(opp_hand, board)
            
            if our_rank > opp_rank:
                wins += 1
            elif our_rank == opp_rank:
                wins += 0.5
                
        return wins / n_samples

    def calculate_potential(
        self,
        our_hand: np.ndarray,
        board: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[float, float]:
        """
        Calculate positive and negative potential through Monte Carlo simulation.
        
        Parameters
        ----------
        our_hand : np.ndarray
            Our hole cards
        board : np.ndarray
            Current board cards
        n_samples : int
            Number of Monte Carlo samples
            
        Returns
        -------
        Tuple[float, float]
            (PPot, NPot) - Probabilities of improving and deteriorating
        """
        # Get current hand strength
        current_hs = self.calculate_hand_strength(our_hand, board)
        
        # Cards we can still see
        available_cards = self.get_available_cards(np.concatenate([our_hand, board]))
        
        ahead = behind = tied = 0
        improve_when_behind = stay_behind = 0
        decline_when_ahead = stay_ahead = 0
        
        for _ in range(n_samples):
            # Sample opponent hand
            opp_hand = np.random.choice(available_cards, 2, replace=False)
            remaining_cards = np.array([c for c in available_cards if c not in opp_hand])
            
            # Evaluate current situation
            our_rank = self.evaluate_hand(our_hand, board)
            opp_rank = self.evaluate_hand(opp_hand, board)
            
            if our_rank > opp_rank:
                ahead += 1
            elif our_rank < opp_rank:
                behind += 1
            else:
                tied += 1
                
            # Sample remaining board cards
            cards_to_come = 5 - len(board)
            if cards_to_come > 0:
                future_cards = np.random.choice(remaining_cards, cards_to_come, replace=False)
                future_board = np.concatenate([board, future_cards])
                
                # Evaluate future situation
                our_future_rank = self.evaluate_hand(our_hand, future_board)
                opp_future_rank = self.evaluate_hand(opp_hand, future_board)
                
                if our_rank < opp_rank:  # Currently behind
                    if our_future_rank > opp_future_rank:
                        improve_when_behind += 1
                    else:
                        stay_behind += 1
                elif our_rank > opp_rank:  # Currently ahead
                    if our_future_rank < opp_future_rank:
                        decline_when_ahead += 1
                    else:
                        stay_ahead += 1
        
        # Calculate PPot and NPot
        behind_total = max(1, behind + tied/2)
        ahead_total = max(1, ahead + tied/2)
        
        ppot = improve_when_behind / behind_total if behind_total > 0 else 0
        npot = decline_when_ahead / ahead_total if ahead_total > 0 else 0
        
        return ppot, npot

    def calculate_ehs_squared(
        self,
        our_hand: np.ndarray,
        board: np.ndarray,
        n_samples: int = 100
    ) -> float:
        """
        Calculate E[HS²] which combines immediate hand strength with future potential.
        
        E[HS²] = HS * (1 - NPot) + (1 - HS) * PPot
        
        Parameters
        ----------
        our_hand : np.ndarray
            Our hole cards
        board : np.ndarray
            Current board cards
        n_samples : int
            Number of Monte Carlo samples
            
        Returns
        -------
        float
            E[HS²] value combining current strength and future potential
        """
        hs = self.calculate_hand_strength(our_hand, board, n_samples)
        ppot, npot = self.calculate_potential(our_hand, board, n_samples)
        
        # Calculate E[HS²]
        ehs_squared = hs * (1 - npot) + (1 - hs) * ppot
        
        return ehs_squared

    def get_hand_category(self, ehs_squared: float) -> str:
        """
        Categorize hand based on E[HS²] value.
        
        Parameters
        ----------
        ehs_squared : float
            E[HS²] value
            
        Returns
        -------
        str
            Hand category description
        """
        if ehs_squared >= 0.8:
            return "Monster"
        elif ehs_squared >= 0.7:
            return "Very Strong"
        elif ehs_squared >= 0.6:
            return "Strong"
        elif ehs_squared >= 0.5:
            return "Above Average"
        elif ehs_squared >= 0.4:
            return "Below Average"
        elif ehs_squared >= 0.3:
            return "Weak"
        else:
            return "Very Weak"