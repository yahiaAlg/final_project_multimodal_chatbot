def initialize_board():
    """
    Initializes an empty game board for Tic-Tac-Toe.

    Returns:
        list: A list representing the 3x3 game board with empty spaces.
    """
    return [" "] * 9


def display_board(board):
    """
    Displays the current state of the game board.

    Args:
        board (list): The current state of the game board.
    """
    print(f"{board[0]} | {board[1]} | {board[2]}")
    print("--+---+--")
    print(f"{board[3]} | {board[4]} | {board[5]}")
    print("--+---+--")
    print(f"{board[6]} | {board[7]} | {board[8]}")


def player_move(board, player):
    """
    Prompts the player to make a move and updates the board.

    Args:
        board (list): The current state of the game board.
        player (str): The current player ('X' or 'O').
    """
    move = int(input(f"Player {player}, choose your move (1-9): ")) - 1
    if board[move] == " ":
        board[move] = player
    else:
        print("Invalid move. Try again.")
        player_move(board, player)


def check_win(board, player):
    """
    Checks if the current player has won the game.

    Args:
        board (list): The current state of the game board.
        player (str): The current player ('X' or 'O').

    Returns:
        bool: True if the player has won, otherwise False.
    """
    win_conditions = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),  # horizontal
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),  # vertical
        (0, 4, 8),
        (2, 4, 6),  # diagonal
    ]
    for cond in win_conditions:
        if board[cond[0]] == board[cond[1]] == board[cond[2]] == player:
            print(f"Player {player} wins!")
            return True
    return False


def check_draw(board):
    """
    Checks if the game is a draw.

    Args:
        board (list): The current state of the game board.

    Returns:
        bool: True if the game is a draw, otherwise False.
    """
    if " " not in board:
        print("The game is a draw!")
        return True
    return False


def game_loop():
    """
    The main game loop for playing Tic-Tac-Toe.
    Alternates turns between players and checks for win or draw conditions.
    """
    board = initialize_board()
    current_player = "X"

    while True:
        display_board(board)
        player_move(board, current_player)
        if check_win(board, current_player):
            display_board(board)
            break
        if check_draw(board):
            display_board(board)
            break
        current_player = "O" if current_player == "X" else "X"


def colored_symbol(symbol):
    """
    Returns the symbol with color.

    Args:
        symbol (str): The player's symbol ('X' or 'O').

    Returns:
        str: The colored symbol.
    """
    if symbol == "X":
        return "\033[91mX\033[0m"  # Red
    elif symbol == "O":
        return "\033[94mO\033[0m"  # Blue
    else:
        return symbol


def display_board_colored(board):
    """
    Displays the current state of the game board with colored symbols.

    Args:
        board (list): The current state of the game board.
    """
    print(
        f"{colored_symbol(board[0])} | {colored_symbol(board[1])} | {colored_symbol(board[2])}"
    )
    print("--+---+--")
    print(
        f"{colored_symbol(board[3])} | {colored_symbol(board[4])} | {colored_symbol(board[5])}"
    )
    print("--+---+--")
    print(
        f"{colored_symbol(board[6])} | {colored_symbol(board[7])} | {colored_symbol(board[8])}"
    )


def game_loop_colored():
    """
    The main game loop for playing Tic-Tac-Toe with colored symbols.
    Alternates turns between players and checks for win or draw conditions.
    """
    board = initialize_board()
    current_player = "X"

    while True:
        display_board_colored(board)
        player_move(board, current_player)
        if check_win(board, current_player):
            display_board_colored(board)
            break
        if check_draw(board):
            display_board_colored(board)
            break
        current_player = "O" if current_player == "X" else "X"


# To play the game with colored symbols, simply call game_loop_colored()
if __name__ == "__main__":
    # To play the game with colored symbols, simply call game_loop_colored()
    game_loop_colored()
