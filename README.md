[Read the documentation]()

# 🤖 Poker AI

This repository will contain a best effort open source implementation of a poker AI using the ideas of Counterfactual Regret.

## Prerequisites

This repository assumes Python 3.7 or newer is used.

## Installing

Either install from pypi:
```bash
pip install poker_ai 
```

## Command Line Interface (CLI)

We have a CLI that will be installed when you pip install the package. To get help on any option, just add the `--help` flag when invoking the CLI.

How to get a list of commands that can be run:
```bash
poker_ai --help
``` 

You will need to produce some lookup tables that cluster the various information sets. Here is more information on that:
```bash
poker_ai cluster --help
```

### Test Clustering Configuration
For initial testing and validation, use default parameters:
```bash
poker_ai cluster
```

### Production Clustering Configuration
For final training with optimal parameters (requires significant computational resources):
```bash
poker_ai cluster \
  --low_card_rank 2 \
  --high_card_rank 14 \
  --n_river_clusters 225 \
  --n_turn_clusters 175 \
  --n_flop_clusters 125 \
  --n_simulations_river 125 \
  --n_simulations_turn 87 \
  --n_simulations_flop 62 \
  --save_dir "./final_clustering"
```

poker_ai cluster \
  --low_card_rank 2 \
  --high_card_rank 14 \
  --n_river_clusters 400 \
  --n_turn_clusters 300 \
  --n_flop_clusters 200 \
  --n_simulations_river 400 \
  --n_simulations_turn 300 \
  --n_simulations_flop 200 \
  --save_dir "./final_clustering"




poker_ai cluster \
  --low_card_rank 2 \
  --high_card_rank 14 \
  --n_river_clusters 1000 \
  --n_turn_clusters 750 \
  --n_flop_clusters 500 \
  --n_simulations_river 1000 \
  --n_simulations_turn 750 \
  --n_simulations_flop 500 \
  --save_dir "./final_clustering"



These parameters are optimized based on the Pluribus paper and provide a balance between abstraction quality and computational feasibility.

### CPU Utilization

The clustering process automatically utilizes all available CPU cores (configurable via MAX_WORKERS environment variable) and achieves near 100% CPU utilization during intensive computations through efficient parallel processing.

Expected runtime varies by hardware:
- 10 cores: ~69.4 hours
- 32 cores: ~21.7 hours
- 64 cores: ~10.8 hours
- 80 cores: ~8.7 hours
- 128 cores: ~5.4 hours


How to get information on training an agent:
```bash
poker_ai train start --help
```

How to get information on resuming training:
```bash
poker_ai train resume --help
```

Once you have an agent, and want to play against it, you can do the following:
```bash
poker_ai play --help
```

## Build a Bot

### Cluster Hero Information

In poker, the number of card combinations for one player on the river can exceed 56 billion combinations. In order to make this information tractable, we must group together strategically similar situations. We do this with two types of compression: lossy and lossless compression. Currently we only support a 20 card deck without modification.

```bash
poker_ai cluster
```

You'll save the combinations of public information in a file called card_info_lut.joblib located in your project directory.

### Train your bot

We use MCCFR to learn strategies. The MCCFR algorithm uses iterative self-play to adjust strategy based on regret. 

```bash
poker_ai train start
```

You'll create a folder in your project directory with the learned strategy and configuration files, in case you need to resume later.

### Play your bot

Finally, you can play your bot with the following command:

```bash
poker_ai play
```

You'll create a results.yaml file in ~/.poker/. So be sure to see how you stack up against your bot.
 
## Running tests

We are working hard on testing our components, but contributions here are always welcome. You can run the tests by cloning the code, changing directory to this repositories root directory (i.e `poker_ai/`) and call the python test library `pytest`:
```bash
cd /path/to/poker_ai
pip install pytest
pytest
```

See below on how to run the tests from the docker image.

## Building the docker image

We use a custom docker image for our testing suite. 

You'll need to have computed the pickled card information lookup tables first (the cluster command for poker_ai). We build the images like below, in this case the luts are in './research/blueprint_algo'. First we build the parent image, with all of the dependancies.
```bash
docker build --build-arg LUT_DIR=research/blueprint_algo -f ParentDockerfile -t pokerai .
```

Then we build the test image.
```bash
docker build -t pokeraitest .
```

We then can run the tests with:
```bash
docker run -it pokeraitest pytest 
```

This is just a note for the developers, but we can push the parent image to the registry with the following (please ensure the version tag that comes after the colon is correct). We want to do this because we need various dependancies for the remote tests, and travis builds the `pokeraitest` image with the current git commit that has just been pushed.
```bash
docker tag pokerai pokerai/pokerai:1.0.0rc1
docker push pokerai/pokerai:1.0.0rc1
```

## Building documentation

Documentation is hosted, but you can build it yourself if you wish:
```bash
# Build the documentation.
cd /path/to/poker_ai/docs
make html
cd ./_build/html 
# Run a webserver and navigate to localhost and the port (usually 8000) in your browser.
python -m http.server 
```

## Repository Structure

Below is a rough structure of the codebase. 

```
├── applications   # Larger applications like the state visualiser sever.
├── paper          # Main source of info and documentation :)
├── poker_ai       # Main Python library.
│   ├── ai         # Stub functions for ai algorithms.
│   ├── games      # Implementations of poker games as node based objects that
│   │              # can be traversed in a depth-first recursive manner.
│   ├── poker      # WIP general code for managing a hand of poker.
│   ├── terminal   # Code to play against the AI from your console.
│   └── utils      # Utility code like seed setting.
├── research       # A directory for research/development scripts 
│                  # to help formulate understanding and ideas.
└── test           # Python tests.
    ├── functional # Functional tests that test multiple components 
    │              # together.
    └── unit       # Individual tests for functions and objects.
```

## Code Examples

Here are some assorted examples of things that are being built in this repo.

### State based poker traversal

To perform MCCFR, the core algorithm of poker_ai, we need a class that encodes all of the poker rules, that we can apply an action to which then creates a new game state.

```python
pot = Pot()
players = [
    ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
    for player_i in range(n_players)
]
state = ShortDeckPokerState(players=players)
for action in state.legal_actions:
    new_state: ShortDeckPokerState = state.apply_action(action)
```

### Playing against AI in your terminal

We also have some code to play a round of poker against the AI agents, inside your terminal.

The characters are a little broken when captured in `asciinema`, but you'll get the idea by watching this video below. Results should be better in your actual terminal!

<p align="center">
  <a href="https://asciinema.org/a/331234" target="_blank">
    <img src="https://asciinema.org/a/331234.svg" width="500" />
  </a>
</p>
To invoke the code, either call the `run_terminal_app` method directly from the `poker_ai.terminal.runner` module, or call from python like so:

```bash
cd /path/to/poker_ai/dir
python -m poker_ai.terminal.runner       \
  --agent offline                        \ 
  --pickle_dir ./research/blueprint_algo \
  --strategy_path ./research/blueprint_algo/offline_strategy_285800.gz 
```

### Web visualisation code

We are also working on code to visualise a given instance of the `ShortDeckPokerState`, which looks like this:
<p align="center">
  <img src="https://github.com/fedden/poker_ai-poker-AI/blob/develop/assets/visualisation.png">
</p>

It is so we can visualise the AI as it plays, and also debug particular situations visually. The idea as it stands, is a live web-visualisation server like TensorBoard, so you'll just push your current poker game state, and this will be reflected in the visualisations, so you can see what the agents are doing. 

[_The frontend code is based on this codepen._](https://codepen.io/Rovak/pen/ExYeQar)

Here is an example of how you could plot the poker game state:
```python
from plot import PokerPlot
from poker_ai.games.short_deck.player import ShortDeckPokerPlayer
from poker_ai.games.short_deck.state import ShortDeckPokerState
from poker_ai.poker.pot import Pot


def get_state() -> ShortDeckPokerState:
    """Gets a state to visualise"""
    n_players = 6
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    return ShortDeckPokerState(
        players=players, 
        pickle_dir="../../research/blueprint_algo/"
    )


pp: PokerPlot = PokerPlot()
# If you visit http://localhost:5000/ now you will see an empty table.

# ... later on in the code, as proxy for some code that obtains a new state ...
# Obtain a new state.
state: ShortDeckPokerState = get_state()
# Update the state to be plotted, this is sent via websockets to the frontend.
pp.update_state(state)
# http://localhost:5000/ will now display a table with 6 players.
```

### Playing a game of poker

There are two parts to this repository, the code to manage a game of poker, and the code to train an AI algorithm to play the game of poker. A low level thing to first to is to implement a poker engine class that can manage a game of poker.

The reason the poker engine is implemented is because it is useful to have a well-integrated poker environment available during the development of the AI algorithm, incase there are tweaks that must be made to accomadate things like the history of state or the replay of a scenario during Monte Carlo Counterfactual Regret Minimisation. 

The following code is how one might program a round of poker that is deterministic using the engine. This engine is now the first pass that will be used support self play.

```python
from poker_ai import utils
from poker_ai.ai.dummy import RandomPlayer
from poker_ai.poker.table import PokerTable
from poker_ai.poker.engine import PokerEngine
from poker_ai.poker.pot import Pot

# Seed so things are deterministic.
utils.random.seed(42)

# Some settings for the amount of chips.
initial_chips_amount = 10000
small_blind_amount = 50
big_blind_amount = 100

# Create the pot.
pot = Pot()
# Instanciate six players that will make random moves, make sure 
# they can reference the pot so they can add chips to it.
players = [
    RandomPlayer(
        name=f'player {player_i}',
        initial_chips=initial_chips_amount,
        pot=pot)
    for player_i in range(6)
]
# Create the table with the players on it.
table = PokerTable(players=players, pot=pot)
# Create the engine that will manage the poker game lifecycle.
engine = PokerEngine(
    table=table,
    small_blind=small_blind_amount,
    big_blind=big_blind_amount)
# Play a round of Texas Hold'em Poker!
engine.play_one_round()
```

## Roadmap

The following todo will change dynamically as my understanding of the algorithms and the poker_ai project evolves. 

At first, the goal is to prototype in Python as iteration will be much easier and quicker. Once there is a working prototype, write in a systems level language like C++ and optimise for performance. 

### 1. Game engine iteration.
_Implement a multiplayer working heads up no limit poker game engine to support the self-play._
- [x] Lay down the foundation of game objects (player, card etc).
- [x] Add poker hand evaluation code to the engine.
- [x] Support a player going all in during betting.
- [x] Support a player going all in during payouts.
- [x] Lots of testing for various scenarios to ensure logic is working as expected.

### 2. AI iteration.
_Iterate on the AI algorithms and the integration into the poker engine._
- [x] Integrate the AI strategy to support self-play in the multiplayer poker game engine.
- [x] In the game-engine, allow the replay of any round the current hand to support MCCFR. 
- [x] Implement the creation of the blueprint strategy using Monte Carlo CFR miminisation.
- [x] Add the real-time search for better strategies during the game.

### 3. Game engine iteration.
_Strengthen the game engine with more tests and allow users to see live visualisation of game state._
- [x] Start work on a visualisation server to allow a game state to be displayed. 
- [ ] Triple check that the rules are implemented in the poker engine as described in the supplimentary material.
- [ ] Work through the coverage, adding more tests, can never have enough.

<p align="center">
  <img src="https://github.com/fedden/poker_ai/blob/develop/assets/regret.jpeg">
</p>

## License

The code is provided under the copy-left GPL licence. If you need it under a more permissive license then please contact me at leonfedden at gmail dot com.


TODO:
Project Structure and Modularity

• Consider further decoupling core game logic from UI or terminal rendering components to improve testability and maintainability.

• Centralize configuration (constants, environment settings) into dedicated modules to reduce scattered configuration across files.

Code Quality and Consistency

• Enforce consistent coding styles across Python and JavaScript/ Vue code (e.g., using Black, flake8, ESLint, and Prettier).

• Increase type annotations and inline documentation in key modules (e.g., the poker_ai, AI, and clustering directories) to help with clarity and error detection.

Testing and Continuous Integration

• Expand unit and functional test coverage, particularly on critical components like game evaluation and AI decision-making.

• Integrate a CI/CD pipeline to automatically run tests on every commit, ensuring regressions are caught early.

Documentation and Developer Onboarding

• Augment existing README and contribution files with high-level architectural overviews or diagrams that outline inter-module dependencies.

• Ensure that inline comments and module docs are up-to-date, especially in more complex logic areas.

Performance and Scalability

• Profile and optimize performance hotspots in real-time poker decision-making code; consider caching or optimized algorithms as needed.

• Evaluate whether processing CSV datasets (in visualization) might benefit from a more robust data handling solution (e.g., a lightweight database) if the datasets grow.

Containerization and Deployment

• Review Docker configurations (Dockerfile versus ParentDockerfile) to take advantage of multi-stage builds and potentially consolidate them for faster builds and smaller image sizes.

• Clearly document environment-dependent settings and container configurations for smoother deployments.

Frontend Enhancements

• For the Vue-based frontend, consider updating to newer API patterns (e.g., the Composition API) if not already in use and ensure webpack and ESLint configurations leverage current best practices.

• Improve automated testing (using tools like Jest or Cypress) for the frontend code.

Error Handling and Logging

• Standardize error handling by leveraging centralized logging mechanisms (for instance, using Python's logging module) to produce consistent, actionable logs.

• Ensure that exceptions in critical paths are captured and reported with sufficient context.

Design Patterns and Best Practices

• Analyze areas where design patterns (Strategy, Factory, Observer, etc.) could simplify complex logic, reduce duplication, and improve maintainability.

• Leverage direct property access and minimize unnecessary variable declarations where clarity isn't compromised.