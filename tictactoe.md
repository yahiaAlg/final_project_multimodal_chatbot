### TicTacToe Console App Documentation

#### Introduction

This project aims to build a console-based TicTacToe game in Python, following a functional programming style. The game will support two players, with colored symbols, win/draw detection, and score tracking stored in a JSON file.

---

### Milestone 1: Setup and Basic Game Board

#### Task 1: Setup the Project

- **Description**: Initialize the project structure.
- **TODO**:
  - Create a project directory (e.g., `TicTacToe`).
  - Create a main file (`tictactoe.py`).

#### Task 2: Define the Game Board

- **Function**: `initialize_board()`

  - **Description**: Initializes a 3x3 game board.
  - **TODO**:
    - Create a function that returns a list of 9 empty spaces.

- **Function**: `display_board(board)`
  - **Description**: Displays the current state of the game board.
  - **TODO**:
    - Implement a function to print the game board in a 3x3 grid format.
    - Use ANSI escape codes for colorizing the player symbols ('X' and 'O').

Hints:

- Use `os.system('cls' if os.name == 'nt' else 'clear')` to clear the console.
- Use ANSI escape codes like `\033[91m` for red and `\033[94m` for blue.

---

### Milestone 2: Game Logic

#### Task 3: Player Move Logic

- **Function**: `player_move(board, player)`
  - **Description**: Handles player moves.
  - **TODO**:
    - Implement a function to take player input for their move.
    - Validate the input to ensure it is a number between 1 and 9 and the chosen spot is not already taken.
    - Update the board with the player's move.

Hints:

- Use a loop to keep asking for input until a valid move is entered.
- Use `try...except` to handle invalid inputs.

#### Task 4: Win and Draw Logic

- **Function**: `check_win(board, player)`

  - **Description**: Checks if the player has won.
  - **TODO**:
    - Implement a function to check all possible winning combinations (rows, columns, diagonals).
    - Return `True` if the player has a winning combination, otherwise `False`.

- **Function**: `check_draw(board)`
  - **Description**: Checks if the game is a draw.
  - **TODO**:
    - Implement a function to check if all spots on the board are filled.
    - Return `True` if the board is full and no player has won, otherwise `False`.

Hints:

- Use a list of tuples to represent the winning combinations.
- Loop through the winning combinations to check if the player's symbols match any combination.

---

### Milestone 3: Game Loop and Player Alternation

#### Task 5: Game Loop Logic

- **Function**: `game_loop()`
  - **Description**: Main game loop to alternate between players and check for win/draw conditions.
  - **TODO**:
    - Initialize the game board.
    - Set the starting player (e.g., 'X').
    - Loop until the game ends (a player wins or the game is a draw).
    - Display the board at each step.
    - Handle player moves.
    - Check for win/draw conditions after each move.
    - Alternate between players.

Hints:

- Use a while loop for the game loop.
- Use a conditional statement to switch current players (e.g., `current_player = 'O' if current_player == 'X' else 'X'`).

---

### Milestone 4: Console Coloring

#### Task 6: Color the Current Player Symbol

- **Function**: `colored_symbol(symbol)`
  - **Description**: Returns the symbol with appropriate color.
  - **TODO**:
    - Implement a function to return the colored version of the player's symbol using ANSI escape codes.

Hints:

- For 'X', use red color: `\033[91mX\033[0m`.
- For 'O', use blue color: `\033[94mO\033[0m`.

---

### Milestone 5: Player Management and Score Tracking

#### Task 7: Player Management

- **Function**: `load_scores(file_path)`

  - **Description**: Loads player scores from a JSON file.
  - **TODO**:
    - Implement a function to read player scores from a JSON file.
    - Handle the case where the file does not exist.

- **Function**: `save_scores(scores, file_path)`

---

### Milestone 1: Setup and Basic Game Board

#### Task 2: Define the Game Board

- **Function**: `initialize_board()`
  - **Output Example**:

```plaintext
[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
```

- **Function**: `display_board(board)`
  - **Output Example**:

```plaintext
  |   |
--+---+--
  |   |
--+---+--
  |   |
```

### Milestone 2: Game Logic

#### Task 3: Player Move Logic

- **Function**: `player_move(board, player)`
  - **Output Example**:

```plaintext
Player X, choose your move (1-9): 5
```

- **Updated Board After Move**:

```plaintext
  |   |
--+---+--
  | X |
--+---+--
  |   |
```

#### Task 4: Win and Draw Logic

- **Function**: `check_win(board, player)`
  - **Output Example**:

```plaintext
Player X wins!
```

- **Function**: `check_draw(board)`
  - **Output Example**:

```plaintext
The game is a draw!
```

### Milestone 3: Game Loop and Player Alternation

#### Task 5: Game Loop Logic

- **Function**: `game_loop()`
  - **Output Example of Full Game Session**:

```plaintext
  |   |
--+---+--
  |   |
--+---+--
  |   |

Player X, choose your move (1-9): 1
X |   |
--+---+--
  |   |
--+---+--
  |   |

Player O, choose your move (1-9): 5
X |   |
--+---+--
  | O |
--+---+--
  |   |

Player X, choose your move (1-9): 2
X | X |
--+---+--
  | O |
--+---+--
  |   |

Player O, choose your move (1-9): 3
X | X | O
--+---+--
  | O |
--+---+--
  |   |

Player X, choose your move (1-9): 9
X | X | O
--+---+--
  | O |
--+---+--
  |   | X

Player X wins!
```

### Milestone 4: Console Coloring

#### Task 6: Color the Current Player Symbol

- **Function**: `colored_symbol(symbol)`
  - **Output Example** (with colored symbols):

```plaintext
X |   |
--+---+--
  | O |
--+---+--
  |   |
```

With color:

- **Red for 'X'**: `\033[91mX\033[0m`
- **Blue for 'O'**: `\033[94mO\033[0m`

### Example with Colored Symbols

- **Output Example**:

```plaintext
[91mX[0m |   |
--+---+--
  | [94mO[0m |
--+---+--
  |   |
```

(Note: The above example includes ANSI escape codes for color, which will display as colored text in a supported terminal.)

These examples should help visualize the game board and the outputs at each stage of the project.
