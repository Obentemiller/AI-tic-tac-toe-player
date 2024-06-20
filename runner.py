```
#executador

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

# Função para verificar o estado de vitória no jogo
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

# Função para exibir o tabuleiro
def print_board(board):
    for row in board:
        print(" | ".join([f'{int(x):2d}' if x != 0 else "  " for x in row]))
        print("-" * 13)

# Função para fazer uma jogada
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

# Carregar o modelo treinado
model = load_model('ania.h5')

# Função principal para jogar contra o modelo
def play_game():
    board = np.zeros((3, 3))
    player = 1  # Jogador 1 é a rede neural
    while check_winner(board) == 0 and np.any(board == 0):
        print_board(board)
        if player == 1:
            move = make_move(board, model)
            if move is not None:
                board[move[0], move[1]] = player
                print(f"Rede Neural joga em ({move[0]+1}, {move[1]+1})")
        else:
            # Jogador humano
            while True:
                try:
                    move = input("Digite sua jogada (linha, coluna): ")
                    move = [int(x) - 1 for x in move.split(',')]
                    if board[move[0], move[1]] == 0:
                        board[move[0], move[1]] = player
                        break
                    else:
                        print("Essa posição já está ocupada. Tente novamente.")
                except (ValueError, IndexError):
                    print("Entrada inválida. Digite novamente no formato: linha,coluna (e.g., 1,2)")
        player *= -1

    print_board(board)
    winner = check_winner(board)
    print("Vencedor:", "Ninguém" if winner == 0 else "Você" if winner == -1 else "Rede Neural")

# Jogar o jogo
play_game()

```
