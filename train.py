```
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def check_winner(board):
    for i in range(3):
        if np.all(board[i, :] == 1) or np.all(board[:, i] == 1):
            return 1
        if np.all(board[i, :] == -1) or np.all(board[:, i] == -1):
            return -1
    if np.all(np.diag(board) == 1) or np.all(np.diag(np.fliplr(board)) == 1):
        return 1
    if np.all(np.diag(board) == -1) or np.all(np.diag(np.fliplr(board)) == -1):
        return -1
    return 0

def generate_training_data(num_games):
    X_train = []
    y_train = []
    
    for _ in range(num_games):
        board = np.zeros((3, 3))
        game_history = []
        player = 1 if np.random.rand() < 0.5 else -1  # Randomly choose who starts
        while check_winner(board) == 0 and np.any(board == 0):
            if player == 1:
                available_moves = np.argwhere(board == 0)
                move = available_moves[np.random.choice(len(available_moves))]
                board[move[0], move[1]] = player
            else:
                move = make_opponent_move(board)  # Opponent makes the move
                board[move[0], move[1]] = player
            game_history.append((board.copy(), move))
            player *= -1
        
        winner = check_winner(board)
        for state, move in game_history:
            X_train.append(state.flatten())
            if winner == 0:
                y_train.append(0)
            elif winner == 1:
                y_train.append(1 if board[move[0], move[1]] == 1 else -1)
            else:
                y_train.append(1 if winner == 1 else -1)
    
    return np.array(X_train), np.array(y_train)

def make_opponent_move(board):
    available_moves = np.argwhere(board == 0)
    move = available_moves[np.random.choice(len(available_moves))]
    return move

num_games = 1000000
X_train, y_train = generate_training_data(num_games)

model = Sequential([
    Dense(64, input_shape=(9,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='tanh')
])

model.compile(optimizer=Adam(), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)
model.save('ania.h5')

def make_move(board, model):
    available_moves = np.argwhere(board == 0)
    best_move = None
    best_value = -np.inf
    
    for move in available_moves:
        board_copy = board.copy()
        board_copy[move[0], move[1]] = 1
        value = model.predict(board_copy.flatten().reshape(1, -1))
        if value > best_value:
            best_value = value
            best_move = move
    
    return best_move

board = np.zeros((3, 3))
player = 1
while check_winner(board) == 0 and np.any(board == 0):
    if player == 1:
        move = make_move(board, model)
        if move is not None:
            board[move[0], move[1]] = player
    else:
        move = make_opponent_move(board)  # Opponent makes the move
        board[move[0], move[1]] = player
    
    print(board)
    player *= -1

winner = check_winner(board)
print("Vencedor:", "Ninguém" if winner == 0 else "Jogador 1" if winner == 1 else "Jogador 2")

```
