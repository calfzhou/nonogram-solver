#!/usr/bin/env python3

import argparse
import copy
import enum
import itertools
import sys
import typing

__version__ = '1.0.0'


class CellType(enum.Enum):
    BOX = enum.auto()
    SPACE = enum.auto()


class LineKind(enum.Enum):
    ROW = enum.auto()
    COL = enum.auto()


class Coord(typing.NamedTuple):
    row: int
    col: int


class Line(typing.NamedTuple):
    kind: LineKind
    n: int


class Board:
    def __init__(self, height: int, width: int):
        self._height = height
        self._width = width
        self._cells = [[None] * width for r in range(height)]
        self._confirmed = 0

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    def __getitem__(self, coord: Coord) -> CellType:
        return self._cells[coord.row][coord.col]

    def __setitem__(self, coord: Coord, value: CellType):
        curr = self._cells[coord.row][coord.col]
        if curr == value:
            return
        elif curr is None and value is not None:
            self._confirmed += 1
        elif curr is not None and value is None:
            self._confirmed -= 1

        self._cells[coord.row][coord.col] = value

    def finished(self) -> bool:
        return self._confirmed == self._height * self._width

    def __str__(self):
        return format_board(self)


class NonogramPuzzle(typing.NamedTuple):
    row_clues: typing.Tuple[typing.Tuple[int]]
    col_clues: typing.Tuple[typing.Tuple[int]]
    board: Board

    @property
    def height(self) -> int:
        return len(self.row_clues)

    @property
    def width(self) -> int:
        return len(self.col_clues)


class GuessData(typing.NamedTuple):
    coord: Coord
    board: Board


class ParadoxError(Exception):
    pass


class Block(typing.NamedTuple):
    begin: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.begin

    @classmethod
    def build(cls, begin: int, end: int=None, min_length=1):
        end = (begin + 1) if (end is None) else end
        return cls(begin, end) if (end - begin >= min_length) else None


# class Block:
#     def __init__(self, begin: int, end: int=None):
#         if end is None:
#             end = begin + 1

#         self.begin = begin
#         self.end = end

#     @property
#     def length(self) -> int:
#         return self.end - self.begin

#     @classmethod
#     def build(cls, begin: int, end: int=None, min_length=1):
#         end = (begin + 1) if (end is None) else end
#         return cls(begin, end) if (end - begin >= min_length) else None


class BlockSection:
    ignore_attrs = set('_prev', '_next')

    def __init__(self, begin: int, end: int, min_length=1):
        self._blocks: typing.List[Block] = []
        self._min_length = min_length
        self._prev: BlockSection = None
        self._next: BlockSection = None

        block = Block.build(begin, end, min_length)
        if block:
            self._blocks.append(block)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in cls.ignore_attrs:
                setattr(result, k, None)
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        return result

    @property
    def begin(self) -> int:
        return self._blocks[0].begin

    @property
    def end(self) -> int:
        return self._blocks[-1].end

    @property
    def length(self) -> int:
        return (self.end - self.begin) if self._blocks else 0

    @property
    def continuous(self) -> bool:
        return len(self._blocks) == 1

    def remove_head(self, end: int):
        if end > self.begin:
            self -= Block(self.begin, end)

    def remove_tail(self, begin: int):
        if self.end > begin:
            self -= Block(begin, self.end)

    def _push_prev(self):
        self._prev.remove_tail(begin=self.end - self._min_length - 1)

    def _push_next(self):
        self._next.remove_head(end=self.begin + self._min_length + 1)

    def _set_prev(self, prev: BlockSection):
        self._prev = prev
        self._push_prev()

    def _set_next(self, next: BlockSection):
        self._next = next
        self._push_next()

    @classmethod
    def chain(cls, sections: typing.Iterable[BlockSection]):
        prev: BlockSection = None
        for curr in sections:
            if prev:
                prev._set_next(curr)
                curr._set_prev(prev)

    @classmethod
    def _norm_other(cls, other):
        if isinstance(other, cls):
            return other
        elif isinstance(other, Block):
            return cls(other.begin, other.end)
        else:
            return cls(other.begin, other.begin + 1)

    def __isub__(self, other):
        other = self._norm_other(other)
        if not self._blocks or not other._blocks or other.end <= self.begin or self.end <= other.begin:
            return self

        begin = self.begin
        end = self.end

        self_index = 0
        other_index = 0
        while self_index < len(self._blocks) and other_index < len(other._blocks):
            self_block = self._blocks[self_index]
            other_block = other._blocks[other_index]
            if self_block.end <= other_block.begin:
                self_index += 1
            elif other_block.end <= self_block.begin:
                other_index += 1
            else:
                replacements = [
                    Block.build(self_block.begin, other_block.begin, self._min_length),
                    Block.build(other_block.end, self_block.end, self._min_length),
                ]
                self._blocks[self_index:self_index + 1] = filter(None, replacements)

        if self._prev and self.end < end:
            self._push_prev()

        if self._next and self.begin > begin:
            self._push_next()

        return self

    def __sub__(self, other):
        result = copy.deepcopy(self)
        result -= other
        return result

    def __iand__(self, other):
        other = self._norm_other(other)
        if not self._blocks or not other._blocks or other.end <= self.begin or self.end <= other.begin:
            self._blocks.clear()
            return self

        begin = self.begin
        end = self.end

        self_index = 0
        other_index = 0
        while self_index < len(self._blocks):
            if other_index >= len(other._blocks):
                del self._blocks[self_index]
                continue

            self_block = self._blocks[self_index]
            other_block = other._blocks[other_index]
            if other_block.end <= self_block.begin:
                other_index += 1
            else:
                replacements = [
                    Block.build(max(self_block.begin, other_block.begin),
                                min(self_block.end, other_block.end),
                                self._min_length)
                ]
                self._blocks[self_index:self_index + 1] = filter(None, replacements)

        if self._prev and self.end < end:
            self._push_prev()

        if self._next and self.begin > begin:
            self._push_next()

        return self


    def __and__(self, other):
        result = copy.deepcopy(self)
        result &= other
        return result


class NonogramSolver:
    def __init__(self):
        self.guess_enabled = False

    def solve(self, puzzle: NonogramPuzzle) -> Board:
        board = puzzle.board or Board(puzzle.height, puzzle.width)

        lines: typing.Set[Line] = set()
        lines.update(Line(LineKind.ROW, i) for i in range(board.height))
        lines.update(Line(LineKind.COL, i) for i in range(board.width))

        guesses: typing.List[GuessData] = []

        while not board.finished():
            if not lines:
                if not self.guess_enabled:
                    break

                # guess
                coord = self.choose(board)
                board[coord] = CellType.BOX
                # TODO: copy.deepcopy(board)
                guesses.append(GuessData(coord, board))
                lines.add(Line(LineKind.ROW, coord.row))
                lines.add(Line(LineKind.COL, coord.col))

            try:
                while lines:
                    line = lines.pop()
                    # process this line
            except ParadoxError:
                if guesses:
                    guess: GuessData = guesses.pop()
                    board[guess.coord] = CellType.SPACE
                    lines.add(Line(LineKind.ROW, coord.row))
                    lines.add(Line(LineKind.COL, coord.col))
                else:
                    raise

        return board

    def solve_line(self, hints: typing.Tuple[int], line: typing.List[CellType]):
        pass


def format_board(board: Board, col_fence=0, row_fence=0) -> str:
    fence_line = []
    for col in range(board.width):
        if col > 0 and col_fence > 0 and col % col_fence == 0:
                fence_line.append('+')
        fence_line.append('-')

    fence_line = ''.join(fence_line)

    mapping = { CellType.BOX: '@', CellType.SPACE: '*' }
    lines = []
    for row in range(board.height):
        if row > 0 and row_fence > 0 and row % row_fence == 0:
            lines.append(fence_line)

        line = []
        for col in range(board.width):
            if col > 0 and col_fence > 0 and col % col_fence == 0:
                line.append('|')
            value = board[Coord(row, col)]
            ch = mapping.get(value, '.')
            line.append(ch)

        lines.append(''.join(line))

    return '\n'.join(lines)


def parse_line_clues(text: str) -> typing.Tuple[int]:
    return tuple(int(x) for x in text.split())


def parse_line_content(text: str, n: int) -> typing.List[CellType]:
    mapping = { 'o': CellType.BOX, '@': CellType.BOX, 'x': CellType.SPACE, '*': CellType.SPACE }
    line = [mapping.get(c.lower()) for c in text if c != '|']
    if len(line) < n:
        line.extend(itertools.repeat(None, n - len(line)))
    elif len(line) > n:
        raise ValueError(f'given line content `{text}` is longer than given line length {n}')

    return line


def load_puzzle(file_path):
    row_clues = []
    col_clues = []
    board: Board = None
    with (open(file_path) if file_path else sys.stdin) as f:
        loading = -2
        for line in f:
            line = line.rstrip('\r\n')
            if line.startswith('#') or '-' in line:
                continue

            if loading == -2:
                if line == '':
                    loading += 1
                else:
                    row_clues.append(parse_line_clues(line))
            elif loading == -1:
                if line == '':
                    loading += 1
                else:
                    col_clues.append(parse_line_clues(line))
            elif loading < len(row_clues):
                if line == '':
                    break
                else:
                    board = board or Board(len(row_clues), len(col_clues))
                    row_content = parse_line_content(line, len(col_clues))
                    for col, value in enumerate(row_content):
                        if value:
                            board[Coord(loading, col)] = value

                    loading += 1
            else:
                break

    return NonogramPuzzle(tuple(row_clues), tuple(col_clues), board)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Nonograms Puzzle Solver', allow_abbrev=False)
    parser.add_argument('--version', action='version', version=f'nonograms_solver {__version__}')
    subparsers = parser.add_subparsers(dest='solver', required=True, help='choose a solver')

    parser_g = subparsers.add_parser('gram', help='gram solver')
    parser_g.add_argument('puzzle_file', nargs='?',
                          help='a file contains the nanogram puzzle, see puzzles/*.txt for example'
                          ' (default: read from stdin)')

    parser_l = subparsers.add_parser('line', help='single line solver')
    parser_l.add_argument('length', type=int,
                         help='length of line')
    parser_l.add_argument('clues', nargs='+', metavar='clue',
                          help='clue numbers')
    parser_l.add_argument('--content', default='',
                         help='content of the line, `o` or `@` for box, `x` or `*` for space,'
                         ' `|` for border (optional), other character for unknown (case insensitive)')

    return parser


def create_solver(args) -> NonogramSolver:
    solver = NonogramSolver()
    return solver


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    print(args)

    solver = create_solver(args)
    puzzle = load_puzzle(args.puzzle_file)
    print(puzzle.row_clues)
    print(puzzle.col_clues)
    print(puzzle.board)


if __name__ == '__main__':
    main()
