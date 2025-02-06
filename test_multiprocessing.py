import time
import os
import psutil
import threading
from pathlib import Path
from poker_ai.clustering.card_info_lut_builder import CardInfoLutBuilder

def monitor_cpu():
    while True:
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        print(f"CPU Usage per core: {cpu_percent}")
        print(f"Average CPU Usage: {sum(cpu_percent)/len(cpu_percent):.1f}%")
        time.sleep(1)

def run_clustering():
    # Create test directory
    test_dir = Path("./test_clustering")
    test_dir.mkdir(exist_ok=True)

    # Set number of workers to use all available cores
    os.environ['MAX_WORKERS'] = str(os.cpu_count())
    print(f"Using {os.cpu_count()} cores")

    # Run a small clustering test
    builder = CardInfoLutBuilder(
        n_simulations_river=20,
        n_simulations_turn=20,
        n_simulations_flop=20,
        low_card_rank=10,
        high_card_rank=14,
        save_dir=str(test_dir)
    )

    builder.compute(
        n_river_clusters=20,
        n_turn_clusters=20,
        n_flop_clusters=20
    )

    # Sleep briefly to see final CPU usage
    time.sleep(5)

if __name__ == '__main__':
    # Start CPU monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
    monitor_thread.start()
    
    # Run the clustering
    run_clustering()