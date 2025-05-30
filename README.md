# OrbitoAgent

An AI agent designed to play the digital version of the board game **Orbito**. Orbito is a strategic game similar to Connect Four, but played on a 4x4 board with unique mechanics such as moving opponent pieces and rotating the board after each turn.

## Features

- Implements adversarial search strategies (e.g., Minimax with Alpha-Beta Pruning).
- Supports multiple agent types:
  - `OrbitoAgent`: Basic agent with adversarial search.
  - `OrbitoAgentV1`: Enhanced agent with improved heuristics.
  - `OrbitoAgentV2`: Advanced agent with move ordering and shallow search optimizations.
  - `AgentRandom`: Random move generator.
  - `AgentHuman`: Allows a human player to interact with the game.
- Fully integrated with the Orbito environment for simulation and testing.

### Installation

`pip install -r requirements.txt`


## How to Use

## Running the Game

You can run the Orbito game using the `orbito_runner.py` script:

```bash
# Run a single game between two random agents
python orbito_runner.py -a random -A random

# Run 10 games between OrbitoAgentV1 and OrbitoAgentV2 without rendering
python orbito_runner.py -c 10 -a agent1 -A agent2 -r none

# Play as a human against OrbitoAgent
python orbito_runner.py -a human -A agent
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--episode-count`, `-c` | Number of episodes to run (default: 1) |
| `--logging-level`, `-l` | Logging level: warn, info, debug (default: warn) |
| `--seed` | Seed for the environment's random number generator (default: 0) |
| `--render-mode`, `-r` | Display style: human, none (default: human) |
| `--agent1`, `-a` | Agent for player 1: random, agent1, human, agent, agent2 (default: random) |
| `--agent2`, `-A` | Agent for player 2: random, agent1, human, agent, agent2 (default: random) |

## Playing as a Human

When playing as a human against an AI agent, you will be prompted to input your moves during your turn. The input format is as follows:

1. **Move Opponent's Piece**: You will first be asked to select an opponent's piece to move. Provide the index of the piece you want to move (0-15, where the board is represented as a 1D array in row-major order).
2. **Move Destination**: Next, you will be asked to specify the destination index (0-15) where you want to move the opponent's piece.
3. **Place Your Piece**: After moving the opponent's piece, you will be prompted to select an index (0-15) to place your own piece.
4. **Confirm Rotation**: The board will automatically rotate counterclockwise after your move.

### Example Gameplay

When playing as a human, the board is represented as a 4x4 grid with positions numbered 0-15:

```
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

#### Turn Example:
1. **Moving an opponent's piece**:
    ```
    Move Opponent's Piece: 6
    Move Destination: 5
    ```

2. **Placing your piece**:
    ```
    Place Your Piece: 10
    ```

3. The board will automatically rotate counterclockwise, and the AI will take its turn.

### Tips
- You can only move an opponent's piece to an adjacent empty position
- You must place your piece in an empty position
- Invalid moves will prompt you to try again
- The output will show you your valid moves


## Performance Evaluation

Agents are evaluated based on the following rewards:
- **Win**: 1
- **Draw**: 0
- **Loss**: -1

For a detailed description of the Performance Measure, Environment, Actions, and Sensors, refer to the PEAS Assessment.

![Gameplay Demo](orbito_gameplay.gif)