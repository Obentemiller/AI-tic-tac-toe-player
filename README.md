# AI-tic-tac-toe-player( I couldn't think of a better name )
This program combines supervised learning techniques (generating training data) and reinforcement learning (using the model to make decisions) to train a simple neural network model to play Tic-Tac-Toe effectively against an opponent making random moves. It illustrates how neural networks can be used to learn complex strategies based on input data and reward.

### Explanation of the Trainer Program

This program implements a trainer for a neural network model that learns to play Tic-Tac-Toe against an opponent making random moves. Let's understand how it works step by step:

### 1. `check_winner` Function

The `check_winner(board)` function checks for a winner in the Tic-Tac-Toe game based on the current state of the board (`board`). It returns:
- `1` if Player 1 wins.
- `-1` if Player 2 (opponent) wins.
- `0` if the game is ongoing or ends in a draw.

### 2. Generating Training Data (`generate_training_data`)

The `generate_training_data(num_games)` function generates training data for the neural network model. It does the following:
- Plays `num_games` games from scratch, where each game is played randomly until there is a winner or a draw.
- For each game, it records the state of the board after each move.
- Uses a random strategy to determine who starts (Player 1 or Player 2).
- Stores the final state of the board and whether the game was won by Player 1, Player 2, or ended in a draw.

### 3. `make_opponent_move` Function

The `make_opponent_move(board)` function chooses a random move for the opponent (Player 2) by selecting an empty position on the board (`board`) where no player has made a move yet.

### 4. Building the Neural Network Model

The neural network model is built using TensorFlow Keras:
- It's a sequential neural network with three dense (`Dense`) layers.
- The first layer (`Dense(64, input_shape=(9,), activation='relu')`) takes a flattened 9-element vector (representing the 3x3 board) and uses the ReLU activation function.
- The second layer (`Dense(64, activation='relu')`) has 64 units with ReLU activation.
- The third layer (`Dense(1, activation='tanh')`) has one unit with the hyperbolic tangent activation function (`tanh`), which predicts the value of the board state.

### 5. Compiling and Training the Model

The model is compiled with the Adam optimizer and mean squared error loss function (`mean_squared_error`). The training data (`X_train` and `y_train`) generated earlier are used to train the model with 50 epochs and a batch size of 32.

### 6. `make_move` Function

The `make_move(board, model)` function is used by Player 1 to determine the best move based on the trained model:
- It iterates over all empty positions on the board.
- For each empty position, it creates a copy of the board with the potential move made by Player 1.
- It uses the model to predict the value of this new board state.
- It selects the move that results in the highest predicted value by the model.

### 7. Main Game

- The main game is then played using the `make_move` function for Player 1 and `make_opponent_move` for Player 2.
- The board is displayed after each move.
- The game continues until there is a winner or a draw, determined by the `check_winner` function.




To run the provided program, you need the following libraries installed:

1. **numpy**: For numerical operations and array handling.
2. **tensorflow**: For building and training the neural network model.
3. **matplotlib**: Optional, for plotting and visualization (used in some variations of the program).

You can install these libraries using Python's package manager `pip`. Here are the commands to install them:

```
pip install numpy
pip install tensorflow
pip install matplotlib
```


