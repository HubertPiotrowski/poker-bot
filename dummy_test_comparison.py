import os
import time
import math
import numpy as np
import concurrent.futures
import multiprocessing
import gc
import tqdm

from poker_ai.clustering.enhanced_evaluator import EnhancedEvaluator
EnhancedEvaluator.calculate_ehs_squared = lambda self, our_hand, board, n_samples: 0.5
EnhancedEvaluator.calculate_potential = lambda self, our_hand, board, n_samples: (0.3, 0.2)

from poker_ai.clustering.card_info_lut_builder import CardInfoLutBuilder, process_river_ehs_worker

# Define module-level worker functions for turn and flop.
def process_turn_ehs_worker(public, cards, n_simulations_turn):
    """
    Dummy worker function for processing turn combos.
    Assume turn combo has 6 cards: first 2 are hole cards and next 4 are board.
    For testing, we simply return a fixed vector.
    """
    # In a realistic scenario, you would call similar functions as for river.
    return np.array([0.55, 0.45, 0.0])

def process_flop_ehs_worker(public, cards, n_simulations_flop):
    """
    Dummy worker function for processing flop combos.
    Assume flop combo has 5 cards: first 2 are hole cards and next 3 are board.
    For testing, we simply return a fixed vector.
    """
    return np.array([0.65, 0.25, 0.1])

def old_compute_river_clusters(builder, river):
    """
    Old implementation using ThreadPoolExecutor.
    """
    max_workers = int(os.environ.get('MAX_WORKERS', os.cpu_count()))
    chunk_size = max(1, min(100, len(river) // (max_workers * 4)))
    result = []
    effective_batch_size = chunk_size * max_workers
    total_batches = math.ceil(len(river) / effective_batch_size)
    print(f"Old river version: {len(river)} combos, chunk_size={chunk_size}, max_workers={max_workers}, total_batches={total_batches}")
    for i in tqdm.tqdm(range(0, len(river), effective_batch_size), desc="Old river thread batches", total=total_batches):
        batch = river[i:i+effective_batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(builder.process_river_ehs, batch, chunksize=chunk_size))
        result.extend(batch_results)
        gc.collect()
    return result

def current_compute_river_clusters(builder, river):
    """
    Current implementation using ProcessPoolExecutor.
    """
    max_workers = int(os.environ.get('MAX_WORKERS', os.cpu_count()))
    chunk_size = max(1, min(100, len(river) // (max_workers * 4)))
    from functools import partial
    ctx = multiprocessing.get_context("spawn")
    worker_func = partial(process_river_ehs_worker, cards=builder._cards, n_simulations_river=builder.n_simulations_river)
    result = []
    effective_batch_size = chunk_size * max_workers
    total_batches = math.ceil(len(river) / effective_batch_size)
    print(f"Current river version: {len(river)} combos, chunk_size={chunk_size}, max_workers={max_workers}, total_batches={total_batches}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        for i in tqdm.tqdm(range(0, len(river), effective_batch_size), desc="Current river process batches", total=total_batches):
            batch = river[i:i + effective_batch_size]
            batch_results = list(executor.map(worker_func, batch, chunksize=chunk_size))
            result.extend(batch_results)
            gc.collect()
    return result

def old_compute_turn_clusters(builder, turn):
    """
    Old implementation for turn combos using ThreadPoolExecutor.
    """
    max_workers = int(os.environ.get('MAX_WORKERS', os.cpu_count()))
    chunk_size = max(1, min(100, len(turn) // (max_workers * 4)))
    result = []
    effective_batch_size = chunk_size * max_workers
    total_batches = math.ceil(len(turn) / effective_batch_size)
    print(f"Old turn version: {len(turn)} combos, chunk_size={chunk_size}, max_workers={max_workers}, total_batches={total_batches}")
    for i in tqdm.tqdm(range(0, len(turn), effective_batch_size), desc="Old turn thread batches", total=total_batches):
        batch = turn[i:i+effective_batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Using the instance method for processing turn; bound to builder.
            batch_results = list(executor.map(builder.process_turn_ehs_distributions, batch, chunksize=chunk_size))
        result.extend(batch_results)
        gc.collect()
    return result

def current_compute_turn_clusters(builder, turn):
    """
    Current implementation for turn combos using ProcessPoolExecutor.
    """
    max_workers = int(os.environ.get('MAX_WORKERS', os.cpu_count()))
    chunk_size = max(1, min(100, len(turn) // (max_workers * 4)))
    from functools import partial
    ctx = multiprocessing.get_context("spawn")
    worker_func = partial(process_turn_ehs_worker, cards=builder._cards, n_simulations_turn=builder.n_simulations_turn)
    result = []
    effective_batch_size = chunk_size * max_workers
    total_batches = math.ceil(len(turn) / effective_batch_size)
    print(f"Current turn version: {len(turn)} combos, chunk_size={chunk_size}, max_workers={max_workers}, total_batches={total_batches}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        for i in tqdm.tqdm(range(0, len(turn), effective_batch_size), desc="Current turn process batches", total=total_batches):
            batch = turn[i:i + effective_batch_size]
            batch_results = list(executor.map(worker_func, batch, chunksize=chunk_size))
            result.extend(batch_results)
            gc.collect()
    return result

def old_compute_flop_clusters(builder, flop):
    """
    Old implementation for flop combos using ThreadPoolExecutor.
    """
    max_workers = int(os.environ.get('MAX_WORKERS', os.cpu_count()))
    chunk_size = max(1, min(100, len(flop) // (max_workers * 4)))
    result = []
    effective_batch_size = chunk_size * max_workers
    total_batches = math.ceil(len(flop) / effective_batch_size)
    print(f"Old flop version: {len(flop)} combos, chunk_size={chunk_size}, max_workers={max_workers}, total_batches={total_batches}")
    for i in tqdm.tqdm(range(0, len(flop), effective_batch_size), desc="Old flop thread batches", total=total_batches):
        batch = flop[i:i+effective_batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(builder.process_flop_potential_aware_distributions, batch, chunksize=chunk_size))
        result.extend(batch_results)
        gc.collect()
    return result

def current_compute_flop_clusters(builder, flop):
    """
    Current implementation for flop combos using ProcessPoolExecutor.
    """
    max_workers = int(os.environ.get('MAX_WORKERS', os.cpu_count()))
    chunk_size = max(1, min(100, len(flop) // (max_workers * 4)))
    from functools import partial
    ctx = multiprocessing.get_context("spawn")
    worker_func = partial(process_flop_ehs_worker, cards=builder._cards, n_simulations_flop=builder.n_simulations_flop)
    result = []
    effective_batch_size = chunk_size * max_workers
    total_batches = math.ceil(len(flop) / effective_batch_size)
    print(f"Current flop version: {len(flop)} combos, chunk_size={chunk_size}, max_workers={max_workers}, total_batches={total_batches}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        for i in tqdm.tqdm(range(0, len(flop), effective_batch_size), desc="Current flop process batches", total=total_batches):
            batch = flop[i:i + effective_batch_size]
            batch_results = list(executor.map(worker_func, batch, chunksize=chunk_size))
            result.extend(batch_results)
            gc.collect()
    return result

if __name__ == "__main__":
    # Create dummy data for river, turn, and flop combos.
    # River combos: 7 cards (2 hole + 5 community)
    dummy_river = np.random.randint(0, 52, size=(2000, 7))
    # Turn combos: 6 cards (2 hole + 4 community)
    dummy_turn = np.random.randint(0, 52, size=(2000, 6))
    # Flop combos: 5 cards (2 hole + 3 community)
    dummy_flop = np.random.randint(0, 52, size=(2000, 5))
    
    # Dummy full deck (for EnhancedEvaluator and others)
    dummy_cards = np.arange(52)
    
    # Instantiate builder with dummy parameters; note low_card_rank changed to 10 for speed.
    builder = CardInfoLutBuilder(
        n_simulations_river=5,
        n_simulations_turn=5,
        n_simulations_flop=5,
        low_card_rank=10,
        high_card_rank=14,
        save_dir="./dummy_clustering"
    )
    
    # Override necessary attributes with dummy data.
    builder.river = dummy_river
    builder._cards = dummy_cards
    # For turn and flop, set attributes.
    builder.turn = dummy_turn
    builder.flop = dummy_flop
    
    # Monkey-patch the enhanced evaluator to bypass complex hand evaluations.
    builder.enhanced_evaluator.calculate_ehs_squared = lambda our_hand, board, n_samples: 0.5
    builder.enhanced_evaluator.calculate_potential = lambda our_hand, board, n_samples: (0.3, 0.2)
    # Set dummy centroids to enable turn and flop computations without prior river clustering.
    builder.centroids["river"] = np.array([[0.5, 0.5, 0.0]])
    builder.centroids["turn"] = np.array([[0.6, 0.3, 0.1]])
    
    print("----- Running old (thread-based) river clusters computation -----")
    start_old = time.time()
    old_river_results = old_compute_river_clusters(builder, dummy_river)
    end_old = time.time()
    print(f"Old river version completed in {end_old - start_old:.2f} seconds, processed {len(old_river_results)} combos.")
    
    print("\n----- Running current (process-based) river clusters computation -----")
    start_current = time.time()
    current_river_results = current_compute_river_clusters(builder, dummy_river)
    end_current = time.time()
    print(f"Current river version completed in {end_current - start_current:.2f} seconds, processed {len(current_river_results)} combos.")
    
    print("\n----- Running old (thread-based) turn clusters computation -----")
    start_old_turn = time.time()
    old_turn_results = old_compute_turn_clusters(builder, dummy_turn)
    end_old_turn = time.time()
    print(f"Old turn version completed in {end_old_turn - start_old_turn:.2f} seconds, processed {len(old_turn_results)} combos.")
    
    print("\n----- Running current (process-based) turn clusters computation -----")
    start_current_turn = time.time()
    current_turn_results = current_compute_turn_clusters(builder, dummy_turn)
    end_current_turn = time.time()
    print(f"Current turn version completed in {end_current_turn - start_current_turn:.2f} seconds, processed {len(current_turn_results)} combos.")
    
    print("\n----- Running old (thread-based) flop clusters computation -----")
    start_old_flop = time.time()
    old_flop_results = old_compute_flop_clusters(builder, dummy_flop)
    end_old_flop = time.time()
    print(f"Old flop version completed in {end_old_flop - start_old_flop:.2f} seconds, processed {len(old_flop_results)} combos.")
    
    print("\n----- Running current (process-based) flop clusters computation -----")
    start_current_flop = time.time()
    current_flop_results = current_compute_flop_clusters(builder, dummy_flop)
    end_current_flop = time.time()
    print(f"Current flop version completed in {end_current_flop - start_current_flop:.2f} seconds, processed {len(current_flop_results)} combos.")
    
    # Optionally compare a few results for each stage.
    if len(old_river_results) > 0 and len(current_river_results) > 0:
        index = np.random.randint(0, min(len(old_river_results), len(current_river_results)))
        diff = np.linalg.norm(old_river_results[index] - current_river_results[index])
        print(f"River: Difference at random index {index}: {diff:.4f}")
    if len(old_turn_results) > 0 and len(current_turn_results) > 0:
        index = np.random.randint(0, min(len(old_turn_results), len(current_turn_results)))
        diff = np.linalg.norm(old_turn_results[index] - current_turn_results[index])
        print(f"Turn: Difference at random index {index}: {diff:.4f}")
    if len(old_flop_results) > 0 and len(current_flop_results) > 0:
        index = np.random.randint(0, min(len(old_flop_results), len(current_flop_results)))
        diff = np.linalg.norm(old_flop_results[index] - current_flop_results[index])
        print(f"Flop: Difference at random index {index}: {diff:.4f}")
    else:
        print("No results to compare.")
