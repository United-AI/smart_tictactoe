from math import inf as infinity
from random import choice
import platform
import time
from os import system
import numpy as np
import pickle
import random

"""
An implementation of Minimax AI Algorithm in Tic Tac Toe,
using Python.
This software is available under GPL license.
Author: Clederson Cruz
Year: 2017
License: GNU GENERAL PUBLIC LICENSE (GPL)
"""

HUMAN = 2
COMP = 1
global board
board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

q_table = np.random.uniform(low=-2, high=0, size=([3] * 9 + [9]))
q_table2 = np.random.uniform(low=-2, high=0, size=([3] * 9 + [9]))
turn = 0
LEARNING_RATE = 0.1
DISCOUNT = 0.95  # how important are future rewards
EPISODES = 25000


def evaluate(state):
    """
    Function to heuristic evaluation of state.
    :param state: the state of the current board
    :return: +1 if the computer wins; -1 if the human wins; 0 draw
    """
    if wins(state, COMP):
        score = +1
    elif wins(state, HUMAN):
        score = -1
    else:
        score = 0

    return score


def wins(state, player):
    """
    This function tests if a specific player wins. Possibilities:
    * Three rows    [X X X] or [O O O]
    * Three cols    [X X X] or [O O O]
    * Two diagonals [X X X] or [O O O]
    :param state: the state of the current board
    :param player: a human or a computer
    :return: True if the player wins
    """
    win_state = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]],
    ]
    if [player, player, player] in win_state:
        return True
    else:
        return False


def game_over(state):
    """
    This function test if the human or computer wins
    :param state: the state of the current board
    :return: True if the human or computer wins
    """
    return wins(state, HUMAN) or wins(state, COMP)


def empty_cells(state):
    """
    Each empty cell will be added into cells' list
    :param state: the state of the current board
    :return: a list of empty cells
    """
    cells = []

    for x, row in enumerate(state):
        for y, cell in enumerate(row):
            if cell == 0:
                cells.append([x, y])

    return cells


def valid_move(x, y):
    """
    A move is valid if the chosen cell is empty
    :param x: X coordinate
    :param y: Y coordinate
    :return: True if the board[x][y] is empty
    """
    if [x, y] in empty_cells(board):
        return True
    else:
        return False


def set_move(x, y, player, board=board):
    """
    Set the move on board, if the coordinates are valid
    :param x: X coordinate
    :param y: Y coordinate
    :param player: the current player
    """
    if valid_move(x, y):
        board[x][y] = player
        return True
    else:
        return False


def minimax(state, depth, player):
    """
    AI function that choice the best move
    :param state: current state of the board
    :param depth: node index in the tree (0 <= depth <= 9),
    but never nine in this case (see iaturn() function)
    :param player: an human or a computer
    :return: a list with [the best row, best col, best score]
    """
    if player == COMP:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state):
        x, y = cell[0], cell[1]
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == COMP:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value

    return best


def clean():
    """
    Clears the console
    """
    os_name = platform.system().lower()
    if 'windows' in os_name:
        system('cls')
    else:
        system('clear')


def render(state, c_choice, h_choice):
    """
    Print the board on console
    :param state: current state of the board
    """

    chars = {
        2: h_choice,
        1: c_choice,
        0: ' '
    }
    str_line = '---------------'

    print('\n' + str_line)
    for row in state:
        for cell in row:
            symbol = chars[cell]
            print(f'| {symbol} |', end='')
        print('\n' + str_line)


def reset_board():
    clean()
    global board
    board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]


def human_turn(c_choice, h_choice):
    """
    The Human plays choosing a valid move.
    :param c_choice: computer's choice X or O
    :param h_choice: human's choice X or O
    :return:
    """
    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        return

    # Dictionary of valid moves
    move = -1
    moves = {
        1: [0, 0], 2: [0, 1], 3: [0, 2],
        4: [1, 0], 5: [1, 1], 6: [1, 2],
        7: [2, 0], 8: [2, 1], 9: [2, 2],
    }

    clean()
    print(f'Human turn [{h_choice}]')
    render(board, c_choice, h_choice)

    while move < 1 or move > 9:
        try:
            move = int(input('Use numpad (1..9): '))
            coord = moves[move]
            can_move = set_move(coord[0], coord[1], HUMAN)

            if not can_move:
                print('Bad move')
                move = -1
        except (EOFError, KeyboardInterrupt):
            print('Bye')
            exit()
        except (KeyError, ValueError):
            print('Bad choice')


################## q-learning ##########################

def board_to_q_index(board):
    q_index = []
    for row in board:
        for square in row:
            q_index.append(square)
    return q_index


def numeric_to_graphical_turn(turn):
    return int(turn / 3), turn % 3


def make_move(ai_number):
    if ai_number == 1:
        set_move(x, y, COMP)
    else:
        set_move(x, y, HUMAN)


def ai_turn(c_choice, h_choice, ai_number, board, displayed_game):
    """
    It calls the minimax function if the depth < 9,
    else it choices a random coordinate.
    :param c_choice: computer's choice X or O
    :param h_choice: human's choice X or O
    :return:
    """
    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        print(board)
        return
    clean()
    q_turn(board, ai_number, displayed_game)


def q_turn(board, ai_number, displayed_game):
    global q_table
    q_index = board_to_q_index(board)
    if ai_number == 1:
        decisions_original = \
        q_table[q_index[0]][q_index[1]][q_index[2]][q_index[3]][q_index[4]][q_index[5]][q_index[6]][q_index[7]][
            q_index[8]]
    else:
        decisions_original = \
        q_table2[q_index[0]][q_index[1]][q_index[2]][q_index[3]][q_index[4]][q_index[5]][q_index[6]][q_index[7]][
            q_index[8]]
    decisions = decisions_original.copy()
    while True:
        best_decision = np.argmax(decisions)
        x, y = numeric_to_graphical_turn(best_decision)
        if board[x][y] == 0:
            if displayed_game:
                print(decisions[best_decision])
                print(decisions)
            best_decision_value = decisions_original[best_decision]
            if ai_number == 1:
                set_move(x, y, COMP)
                reward = set_COMP_reward()
            else:
                set_move(x, y, HUMAN)
                reward = set_HUMAN_reward()

            new_q = bellman_equation(best_decision_value, ai_number, reward, displayed_game)
            decisions_original[best_decision] = new_q
            break

        else:
            decisions[best_decision] = -100


def bellman_equation(best_decision_value, ai_number, reward, displayed_game):
    q_index = board_to_q_index(board)
    enemy_max_future_q, enemy_max_future_move = get_enemy_future_moves(ai_number, q_table, q_index)

    get_best_future_move(ai_number, enemy_max_future_move)

    # wenn man gewinnt hat enemy keinen besten Zug
    if reward == 0:
        new_q = (1 - LEARNING_RATE) * best_decision_value + LEARNING_RATE * reward

    else:
        new_q = (1 - LEARNING_RATE) * best_decision_value + LEARNING_RATE * (
                    reward + DISCOUNT * (-2 - enemy_max_future_q))
    if displayed_game:
        print(f"0.9*{best_decision_value}+0.1*({reward}+0.95*(-2-{enemy_max_future_q}))")
        print(f"current q: {best_decision_value}, new q: {new_q}")
    return new_q


def get_best_future_move(ai_number, enemy_max_future_move):
    print("max future move", enemy_max_future_move)
    x, y = numeric_to_graphical_turn(enemy_max_future_move)
    global board
    board_copy = board.copy()
    print(board_copy)
    if ai_number != 1:
        set_move(x, y, COMP, board_copy)
    else:
        set_move(x, y, HUMAN, board_copy)
    print(board_copy)


def get_enemy_future_moves(ai_number, q_table, q_index):
    if ai_number == 1:
        enemy_decisions = \
        q_table2[q_index[0]][q_index[1]][q_index[2]][q_index[3]][q_index[4]][q_index[5]][q_index[6]][q_index[7]][
            q_index[8]]
    else:
        enemy_decisions = \
        q_table[q_index[0]][q_index[1]][q_index[2]][q_index[3]][q_index[4]][q_index[5]][q_index[6]][q_index[7]][
            q_index[8]]
    possible_enemy_decisions = get_choices(enemy_decisions.copy())
    enemy_max_future_q = np.max(possible_enemy_decisions)
    enemy_max_future_index = np.argmax(possible_enemy_decisions)
    return enemy_max_future_q, enemy_max_future_index


def get_choices(decisions):
    decisions = list(decisions)
    for i, decision in enumerate(decisions):
        x, y = numeric_to_graphical_turn(i)
        if board[x][y] != 0:
            decisions[i] = "nan"
    decisions = [x for x in decisions if x != 'nan']
    if not decisions:
        decisions = [-1]
    return decisions


def set_COMP_reward():
    if wins(board, COMP):
        reward = 10
    elif wins(board, HUMAN):
        reward = -2
    else:
        reward = random.uniform(-0.9, -1.1)
    return reward


def set_HUMAN_reward():
    if wins(board, HUMAN):
        reward = 10
    elif wins(board, COMP):
        reward = -2
    else:
        reward = random.uniform(-0.9, -1.1)
    return reward


def main():
    """
    Main function that calls all functions
    """
    clean()
    h_choice = ''  # X or O
    c_choice = ''  # X or O
    first = ''  # if human is the first

    # Human chooses X or O to play
    while h_choice != 'O' and h_choice != 'X':
        try:
            print('')
            h_choice = input('Choose X or O\nChosen: ').upper()
        except (EOFError, KeyboardInterrupt):
            print('Bye')
            exit()
        except (KeyError, ValueError):
            print('Bad choice')

    # Setting computer's choice
    if h_choice == 'X':
        c_choice = 'O'
    else:
        c_choice = 'X'

    # Human may starts first
    clean()
    while first != 'Y' and first != 'N':
        try:
            first = input('First to start?[y/n]: ').upper()
        except (EOFError, KeyboardInterrupt):
            print('Bye')
            exit()
        except (KeyError, ValueError):
            print('Bad choice')

    # Main loop of this game
    while len(empty_cells(board)) > 0 and not game_over(board):
        if first == 'N':
            ai_turn(c_choice, h_choice)
            first = ''

        human_turn(c_choice, h_choice)
        ai_turn(c_choice, h_choice)

    # Game over message
    if wins(board, HUMAN):
        clean()
        print(f'Human turn [{h_choice}]')
        render(board, c_choice, h_choice)
        print('YOU WIN!')
    elif wins(board, COMP):
        clean()
        print(f'Computer turn [{c_choice}]')
        render(board, c_choice, h_choice)
        print('YOU LOSE!')
    else:
        clean()
        render(board, c_choice, h_choice)
        print('DRAW!')

    exit()


def ai_game():
    games = 10000
    render_every = 200
    c_choice = "X"
    h_choice = "O"
    for x in range(games):
        reset_board()
        player = -1  # 1==Comp, 2==HUMAN
        displayed_game = (x % render_every == 0)
        while len(empty_cells(board)) > 0 and not game_over(board):
            ai_turn(c_choice, h_choice, player, board, displayed_game)
            if displayed_game:
                print(f"Game: {x}")
                render(board, c_choice, h_choice)
                print("\n")
            player *= -1
    with open('ai1.pickle', 'wb') as handle:
        pickle.dump(q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('ai2.pickle', 'wb') as handle:
        pickle.dump(q_table2, handle, protocol=pickle.HIGHEST_PROTOCOL)


ai_game()