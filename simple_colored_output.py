def colored_symbol(symbol):
    """Return the symbol with appropriate color using ANSI escape codes."""
    if symbol == "X":
        return "\033[91mX\033[0m"  # Red for X
    elif symbol == "O":
        return "\033[94mO\033[0m"  # Blue for O
    else:
        return symbol


import os


def initialize_board():
    """Initialize a 3x3 game board with empty spaces."""
    return [" " for _ in range(9)]


def display_board(board):
    """Display the current state of the game board with colored symbols."""
    os.system("cls" if os.name == "nt" else "clear")  # Clears the console
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


# Example usage
board = initialize_board()
board[0] = "X"
board[4] = "O"
board[8] = "X"
display_board(board)
