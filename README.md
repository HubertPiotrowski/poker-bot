# Project Title

This project implements a poker AI that utilizes clustering techniques to organize and evaluate card combinations. The current implementation is designed to precompute lookup tables for different streets (flop, turn, and river) using simulation and clustering methods. The system leverages both thread-based and process-based parallelism for efficient computation.

## Features

- Computation of pre-flop abstractions
- Clustering of river, turn, and flop card combinations using simulation data
- Parallel computation using ProcessPoolExecutor for full CPU utilization
- Lookup table creation for efficient evaluation

## Installation

Instructions to install the project and its dependencies.

## Usage

Details on how to run the poker AI, including command-line scripts and configuration.

## TODO

- [ ] **Blueprint Strategy:** Implement blueprint strategy computation via self-play (inspired by Pluribus) to generate a baseline strategy.
- [ ] **Limited-Lookahead Search:** Develop a limited-lookahead search algorithm that considers a few continuation strategies for each player, enabling dynamic in-game decision adjustments.
- [ ] **Mixed Strategies:** Integrate mixed strategies such as variable bet sizing and donk betting to prevent predictability.
- [ ] **Real-Time Adaptation:** Incorporate real-time decision adjustments based on dynamic clustering outcomes during gameplay.

## License

Details about the project's license.
