import tkinter as tk

# ROWS AND COLUMNS CONSTANTS
ROWS = 3
COLS = 3


class TicTacToe:
    def __init__(self, master):
        self.master = master
        self.board = []  # initialize the board as an empty list of length ROWS*COLUMNS
        for i in range(ROWS):
            row = []
            for j in range(COLS):
                cell = tk.Button(
                    self.master, text=" ", command=lambda x=i, y=j: self.play_move(x, y)
                )  # create a button with a lambda function as the command to play move at (i,j) position
                cell.grid(row=i + 1, column=j)
                row.append(cell)
            self.board.append(row)

        # initialize game variables
        self.turn = "X"
        self.game_over = False
        self.winner = None

    def play_move(self, x, y):
        if not self.game_over:
            button = self.board[x][y]  # get the button at position (i,j)
            if button["text"] == " ":  # check if it's empty
                button["text"] = (
                    self.turn
                )  # set the text to current player's mark (X or O)
                self.check_win()  # check for a winner and update game status
                if not self.game_over:  # if no winner yet, toggle turn and play again
                    self.turn = "O" if self.turn == "X" else "X"
                    button["command"] = lambda x=x, y=y: self.play_move(x, y)

    def check_win(self):
        # horizontal wins
        for i in range(ROWS):
            if all([cell["text"] == "X" for cell in self.board[i]]) or all(
                [cell["text"] == "O" for cell in self.board[i]]
            ):
                self.winner = self.board[i][0][
                    "text"
                ]  # set the winner as the mark of the row
                self.game_over = True

        # vertical wins
        for j in range(COLS):
            if all([self.board[i][j]["text"] == "X" for i in range(ROWS)] or all([cell["text"] == "O" for cell in [row[j] for row in self.board]])):  # type: ignore
                self.winner = self.board[0][j][
                    "text"
                ]  # set the winner as the mark of the column
                self.game_over = True

        # diagonal wins
        if all([self.board[i][i]["text"] == "X" for i in range(ROWS)]) or all(
            [self.board[i][COLS - 1 - i]["text"] == "X" for i in range(ROWS)]
        ):
            self.winner = "X"
            self.game_over = True

        if all(
            [cell["text"] == "O" for row in self.board for cell in row]
        ):  # check if board is full and no winner yet
            self.game_over = True
            self.winner = "Tie"

    def reset(self):
        for row in self.board:
            for button in row:
                button["text"] = " "  # set all buttons to empty
        self.turn = "X"  # start a new game with X's turn
        self.game_over = False
        self.winner = None


root = tk.Tk()
ttt = TicTacToe(root)
root.mainloop()  # run the GUI main loop
