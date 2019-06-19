# nonogram-solver

A deductive solver for [nonograms](https://en.wikipedia.org/wiki/Nonogram).

## Usage

There are two solving modes:

- **gram** mode solves a whole nonogram

- **line** mode solves a single line

### Gram Mode

```
$ ./nonogram_solver.py gram -h
usage: nonogram_solver.py gram [-h] [--guess [{True,False}]]
                               [--show-progress [{True,False}]]
                               [--progress-pause PROGRESS_PAUSE]
                               [--show-deduce [{True,False}]]
                               [--show-guess [{True,False}]]
                               [--grid [WIDTH[,HEIGHT]]]
                               [--line-fence LINE_FENCE]
                               [--full-width [{True,False}]]
                               [puzzle_file]

positional arguments:
  puzzle_file           a file contains the nanogram puzzle, see puzzles/*.txt
                        for example (default: read from stdin)

optional arguments:
  -h, --help            show this help message and exit
  --guess [{True,False}]
                        whether enable guess when puzzle cannot be solved by
                        deducing (default: false)
  --show-progress [{True,False}]
                        whether print board after each deducing step
                        (highlight changes) (default: false)
  --progress-pause PROGRESS_PAUSE
                        pause some time (in seconds) between each progress
                        board view (default: 0.2)
  --show-deduce [{True,False}]
                        whether print every line deducing result (default:
                        false)
  --show-guess [{True,False}]
                        whether print every guessing step (default: false)
  --grid [WIDTH[,HEIGHT]]
                        show major grid line when printing gram with the given
                        size (default: 5,5)
  --line-fence LINE_FENCE
                        if greater than 0, print fence when printing single
                        line (default: 5)
  --full-width [{True,False}]
                        whether use full width char when print gram (default:
                        true)
```

### Line Mode

```
$ ./nonogram_solver.py line -h
usage: nonogram_solver.py line [-h] [--content CONTENT]
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
  --line-fence LINE_FENCE
                        if greater than 0, print fence when printing single
                        line (default: 5)
```
