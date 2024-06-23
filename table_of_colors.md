table of some common colors you can use with ANSI escape codes for text in the console. These codes can be used to colorize text output in supported terminals.

### ANSI Escape Codes for Colors

| Color          | Foreground Code | Background Code |
| -------------- | --------------- | --------------- |
| Black          | `\033[30m`      | `\033[40m`      |
| Red            | `\033[31m`      | `\033[41m`      |
| Green          | `\033[32m`      | `\033[42m`      |
| Yellow         | `\033[33m`      | `\033[43m`      |
| Blue           | `\033[34m`      | `\033[44m`      |
| Magenta        | `\033[35m`      | `\033[45m`      |
| Cyan           | `\033[36m`      | `\033[46m`      |
| White          | `\033[37m`      | `\033[47m`      |
| Bright Black   | `\033[90m`      | `\033[100m`     |
| Bright Red     | `\033[91m`      | `\033[101m`     |
| Bright Green   | `\033[92m`      | `\033[102m`     |
| Bright Yellow  | `\033[93m`      | `\033[103m`     |
| Bright Blue    | `\033[94m`      | `\033[104m`     |
| Bright Magenta | `\033[95m`      | `\033[105m`     |
| Bright Cyan    | `\033[96m`      | `\033[106m`     |
| Bright White   | `\033[97m`      | `\033[107m`     |

### Example Usage

Here's a Python function that demonstrates how to use these ANSI escape codes to colorize text:

```python
def colored_text(text, color_code):
    """Return the text string wrapped in the specified ANSI color code."""
    return f"{color_code}{text}\033[0m"

# Example usage
print(colored_text("This is red text", "\033[31m"))
print(colored_text("This is green text", "\033[32m"))
print(colored_text("This is blue text", "\033[34m"))
print(colored_text("This is bright yellow text", "\033[93m"))
```

### Explanation

- **Foreground Code**: Changes the color of the text.
- **Background Code**: Changes the color of the background behind the text.
- **`\033[0m`**: Resets the text color to default.

### Example with Background Color

To use background colors, you can modify the function to accept both foreground and background color codes:

```python
def colored_text(text, fg_color_code, bg_color_code=None):
    """Return the text string wrapped in the specified ANSI color codes."""
    if bg_color_code:
        return f"{fg_color_code}{bg_color_code}{text}\033[0m"
    else:
        return f"{fg_color_code}{text}\033[0m"

# Example usage
print(colored_text("Text on red background", "\033[37m", "\033[41m"))  # White text on red background
print(colored_text("Text on blue background", "\033[37m", "\033[44m"))  # White text on blue background
print(colored_text("Red text on green background", "\033[31m", "\033[42m"))  # Red text on green background
```

This should give you a good starting point for using ANSI escape codes to colorize text in your console applications.
