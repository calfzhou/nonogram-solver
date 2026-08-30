# nonogram-solver

A deductive solver for [nonograms](https://en.wikipedia.org/wiki/Nonogram).

## Usage

There are two solving modes:

- **gram** mode solves a whole nonogram
- **line** mode solves a single line

### Gram Mode

```shell-session
$ ./nonogram_solver.py gram -h
usage: nonogram_solver.py gram [-h] [--guess | --no-guess]
                               [--show-progress | --no-show-progress]
                               [--progress-pause PROGRESS_PAUSE]
                               [--show-deduce | --no-show-deduce]
                               [--show-guess | --no-show-guess]
                               [--grid [WIDTH[,HEIGHT]]]
                               [--line-fence LINE_FENCE]
                               [--full-width | --no-full-width]
                               [puzzle_file]

positional arguments:
  puzzle_file           a file contains the nanogram puzzle, see puzzles/*.txt
                        for example (default: read from stdin)

options:
  -h, --help            show this help message and exit
  --guess, --no-guess   whether enable guess when puzzle cannot be solved by
                        deducing (default: False)
  --show-progress, --no-show-progress
                        whether print board after each deducing step
                        (highlight changes) (default: False)
  --progress-pause PROGRESS_PAUSE
                        pause some time (in seconds) between each progress
                        board view (default: 0.2)
  --show-deduce, --no-show-deduce
                        whether print every line deducing result (default:
                        False)
  --show-guess, --no-show-guess
                        whether print every guessing step (default: False)
  --grid [WIDTH[,HEIGHT]]
                        show major grid line when printing gram with the given
                        size (default: 5,5)
  --line-fence LINE_FENCE
                        if greater than 0, print fence when printing single
                        line (default: 5)
  --full-width, --no-full-width
                        whether use full width char when print gram (default:
                        True)

### Line Mode

```shell-session
$ ./nonogram_solver.py line -h
usage: nonogram_solver.py line [-h] [--content CONTENT]
                               [--exact | --no-exact]
                               [--line-fence LINE_FENCE]
                               length clue [clue ...]

positional arguments:
  length                length of line
  clue                  clue numbers

optional arguments:
  -h, --help            show this help message and exit
  --content CONTENT     content of the line, `o` or `@` for box, `x` or `*`
                        for space, `|` for border (optional), other character
                        for unknown (case insensitive)
  --exact, --no-exact   whether use exact clue-placement deduction (default:
                        True)
  --line-fence LINE_FENCE
                        if greater than 0, print fence when printing single
                        line (default: 5)
```

## Development & Testing

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The solver
itself uses only the standard library; the test suite needs `ddt` and `pyyaml`,
which are declared in the `dev` dependency group in `pyproject.toml`.

```sh
# Install test dependencies into .venv (also generates uv.lock)
uv sync

# Run the test suite
uv run python -m unittest test_gram test_line
```
