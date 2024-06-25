def colorise_string(string: str) -> str:
    from random import randint

    _color_list = [
        "\033[30m",
        "\033[31m",
        "\033[32m",
        "\033[33m",
        "\033[34m",
        "\033[35m",
        "\033[36m",
        "\033[37m",
        "\033[90m",
        "\033[91m",
        "\033[92m",
        "\033[93m",
        "\033[94m",
        "\033[95m",
        "\033[96m",
        "\033[97m",
    ]
    return "".join(
        [
            _color_list[randint(1, len(_color_list)) - 1] + letter + "\033[0m"
            for letter in string
        ]
    )


if __name__ == "__main__":
    sentence = input("insert something:\n")
    colored_sentence = colorise_string(sentence)
    # printing the colored sentence
    print(colored_sentence)
