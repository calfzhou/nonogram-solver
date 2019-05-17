#!/usr/bin/env python3

import argparse
import copy
import enum
import itertools
import math
import sys
import typing

__version__ = '1.0.0'


class CellType(enum.Enum):
    BOX = enum.auto()
    SPACE = enum.auto()

    def __repr__(self):
        if self == self.BOX:
            return 'BOX'
        elif self == self.SPACE:
            return 'SPACE'


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

    def iter(self):
        return range(self.begin, self.end)

    def __repr__(self):
        return f'[{self.begin + 1},{self.end}]'

    @classmethod
    def build(cls, begin: int, end: int=None, min_length=1):
        end = (begin + 1) if (end is None) else end
        return cls(begin, end) if (end - begin >= min_length) else None


class BlockSection:
    ignore_attrs = {'_prev', '_next'}

    def __init__(self, begin: int, end: int=None, min_length=1):
        self._blocks: typing.List[Block] = []
        self._min_length = min_length

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
    def cell_count(self) -> int:
        return sum(b.length for b in self._blocks)

    def iter(self):
        for block in self._blocks:
            for i in block.iter():
                yield i

    @property
    def continuous(self) -> bool:
        return len(self._blocks) == 1

    def __repr__(self):
        blocks = ', '.join(str(b) for b in self._blocks)
        return f'<{blocks}>'

    @classmethod
    def _norm_other(cls, other):
        if isinstance(other, cls):
            return other
        elif isinstance(other, Block):
            return cls(other.begin, other.end)
        else:
            return cls(other)

    def __contains__(self, other):
        other = self._norm_other(other)
        if not self._blocks or not other._blocks or other.end <= self.begin or self.end <= other.begin:
            return False

        intersect = other & self
        return intersect.cell_count == other.cell_count

    def __isub__(self, other):
        other = self._norm_other(other)
        if not self._blocks or not other._blocks or other.end <= self.begin or self.end <= other.begin:
            return self

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
                intersect = Block.build(max(self_block.begin, other_block.begin),
                                min(self_block.end, other_block.end),
                                self._min_length)
                tail = Block.build(other_block.end, self_block.end, self._min_length)
                self._blocks[self_index:self_index + 1] = filter(None, (intersect, tail))
                if intersect:
                    self_index += 1

        return self

    def __and__(self, other):
        result = copy.deepcopy(self)
        result &= other
        return result


class Ternary(enum.Enum):
    NO = enum.auto()
    YES = enum.auto()
    EXCLUSIVE = enum.auto()


class ClueExtra:
    def __init__(self, index: int, value: int, begin: int, end: int):
        self._index = index
        self._value = value
        self._candidates = BlockSection(begin, end, min_length=value)
        self._boxes: Block = None
        self._prev: ClueExtra = None
        self._next: ClueExtra = None

        self._check_box()

    @property
    def index(self) -> int:
        return self._index

    @property
    def value(self) -> int:
        return self._value

    @property
    def candidates(self) -> BlockSection:
        if self._candidates.length == 0:
            raise ParadoxError(f'clue {self._index + 1} (value {self._value}) has no box candidates')
        return self._candidates

    @property
    def boxes(self) -> Block:
        return self._boxes

    def finished(self) -> bool:
        return self._boxes and self._boxes.length == self._value

    def __repr__(self):
        return f'clue #{self._index + 1} ({self._value}) {self._boxes}'

    @classmethod
    def chain(cls, clues: typing.Iterable):
        prev: ClueExtra = None
        for curr in clues:
            if prev:
                prev._set_next(curr)
                curr._set_prev(prev)
            prev = curr

    def off_chain(self):
        if self._prev:
            self._prev._set_next(None)
            self._set_prev(None)

        if self._next:
            self._next._set_prev(None)
            self._set_next(None)

    def confirm_boxes(self, boxes: Block):
        if self._boxes is None:
            self._boxes = boxes
        elif self._boxes.begin != boxes.begin or self._boxes.end != boxes.end:
            self._boxes = Block(
                min(self._boxes.begin, boxes.begin),
                max(self._boxes.end, boxes.end)
            )
        else:
            return

        self._on_boxes_extended()

    def can_contain_box(self, boxes: Block, boundary: Block) -> Ternary:
        if boxes.length > self._value:
            return Ternary.NO

        if boundary.length < self._value:
            return Ternary.NO

        if boxes not in self._candidates:
            return Ternary.NO

        if (self._candidates & boundary).length == 0:
            return Ternary.NO

        if not self._boxes:
            return Ternary.YES

        merged = Block(
            min(self._boxes.begin, boxes.begin),
            max(self._boxes.end, boxes.end)
        )
        if merged.length <= self._boxes.length + boxes.length:
            return Ternary.EXCLUSIVE

        if merged.length > self._value:
            return Ternary.NO

        # This is not possible.
        # if merged not in self._candidates:
        #     return Ternary.NO

        return Ternary.YES

    def _set_prev(self, prev: BlockSection):
        self._prev = prev
        if self._prev:
            self._push_prev()

    def _set_next(self, next: BlockSection):
        self._next = next
        if self._next:
            self._push_next()

    def _push_prev(self):
        self._prev.remove_tail_candidates(begin=self.candidates.end - self._value - 1)

    def _push_next(self):
        self._next.remove_head_candidates(end=self.candidates.begin + self._value + 1)

    def remove_head_candidates(self, end: int):
        if end > self.candidates.begin:
            self.remove_candidates(Block(self.candidates.begin, end))

    def remove_tail_candidates(self, begin: int):
        if self.candidates.end > begin:
            self.remove_candidates(Block(begin, self.candidates.end))

    def remove_candidates(self, other):
        begin = self.candidates.begin
        end = self.candidates.end
        self._candidates -= other
        if self.candidates.begin > begin or self.candidates.end < end:
            self._on_candidates_removed(begin, end)

    def limit_candidates(self, other):
        begin = self.candidates.begin
        end = self.candidates.end
        self._candidates &= other
        if self.candidates.begin > begin or self.candidates.end < end:
            self._on_candidates_removed(begin, end)

    def _on_candidates_removed(self, old_begin: int, old_end: int):
        if self._prev and self.candidates.end < old_end:
            self._push_prev()

        if self._next and self.candidates.begin > old_begin:
            self._push_next()

        self._check_box()

    def _on_boxes_extended(self):
        padding = self._value - self._boxes.length
        possible = Block(self._boxes.begin - padding, self._boxes.end + padding)
        self.limit_candidates(possible)

    def _check_box(self):
        padding = self._candidates.length - self._value
        if padding >= self._value:
            return

        boxes = Block(self.candidates.begin + padding, self.candidates.end - padding)
        self.confirm_boxes(boxes)


class BoxBlockAndClues(typing.NamedTuple):
    block: Block
    clues: typing.List[ClueExtra]


class LineSolver:
    def __init__(self, clues: typing.Tuple[int], content: typing.List[CellType]):
        self._clues = clues
        self._content = content
        self._width = len(content)

        self._remain_cell = Block(0, self._width)
        self._remain_clue = Block(0, len(self._clues))
        self._clues_ex = None

        self._changes = set()

    @property
    def changes(self) -> typing.Set[int]:
        return self._changes

    def solve(self):
        # Check if already or almost finished.
        if self._check_finish():
            return

        # Trim finished head and tail.
        begin, clue_begin = self._trim_finished(
            self._remain_cell.begin, self._remain_cell.end, self._remain_clue.begin, self._remain_clue.end, step=1)
        end, clue_end = self._trim_finished(
            self._remain_cell.end - 1, begin - 1, self._remain_clue.end - 1, clue_begin - 1, step=-1)
        end += 1
        clue_end += 1

        self._remain_cell = Block(begin, end)
        self._remain_clue = Block(clue_begin, clue_end)

        # Check if all clues finished (remaining spaces only).
        if self._remain_clue.length == 0:
            for i in self._remain_cell.iter():
                self._mark_cell(i, CellType.SPACE)
            return

        # Build clues' block section.
        self._clues_ex = [ClueExtra(j, self._clues[j], begin, end) for j in self._remain_clue.iter()]
        ClueExtra.chain(self._clues_ex)

        # Process known spaces.
        for i in self._remain_cell.iter():
            if self._content[i] == CellType.SPACE:
                for clue in self._clues_ex:
                    clue.remove_candidates(i)

        # Process known boxes.
        known_boxes = self._get_known_boxes()
        finished = False
        while known_boxes and not finished:
            finished = not self._handle_known_boxes(known_boxes)

        # Finalize.
        self._mark_boxes()
        self._mark_spaces()

    def _mark_cell(self, index: int, value: CellType):
        if index < 0 or index >= self._width:
            raise ParadoxError(f'cell {index + 1} index out of range')
        elif self._content[index] is None:
            self._content[index] = value
            self._changes.add(index)
        elif self._content[index] != value:
            raise ParadoxError(f'cell {index + 1} cannot be {self._content[index]}')

    def _check_finish(self) -> bool:
        clue_sum = sum(self._clues)
        if clue_sum + len(self._clues) - 1 > self._width:
            raise ParadoxError('clue sum is too big')

        box_count = 0
        space_count = 0
        for value in self._content:
            if value == CellType.BOX:
                box_count += 1
            elif value == CellType.SPACE:
                space_count += 1

        # Check if all cells finished.
        if box_count + space_count == self._width:
            return True

        if box_count == clue_sum:
            # All boxes finished (or empty line), need fill in spaces.
            value = CellType.SPACE
        elif space_count == self._width - clue_sum:
            # All spaces finished (or full line), need fill in boxes.
            value = CellType.BOX
        else:
            return False

        for i in range(self._width):
            if self._content[i] is None:
                self._mark_cell(i, value)

        return True

    def _trim_finished(self, begin: int, end: int, clue_begin: int, clue_end: int, step: int):
        while begin != end:
            if self._content[begin] is None:
                break
            elif self._content[begin] == CellType.SPACE:
                begin += step
            elif clue_begin == clue_end:
                raise ParadoxError(f'cell {begin + 1} cannot be {CellType.BOX}')
            else:
                begin += step
                for _ in range(self._clues[clue_begin] - 1):
                    self._mark_cell(begin, CellType.BOX)
                    begin += step

                clue_begin += step
                if begin != end:
                    self._mark_cell(begin, CellType.SPACE)
                    begin += step

        return begin, clue_begin

    def _is_space(self, index: int) -> bool:
        if index < self._remain_cell.begin or index >= self._remain_cell.end:
            return True

        value = self._content[index]
        if value == CellType.SPACE:
            return True
        elif value == CellType.BOX:
            return False
        else:
            return not any(index in clue.candidates for clue in self._clues_ex)

    def _find_boundary(self, boxes: Block) -> Block:
        begin = boxes.begin - 1
        while not self._is_space(begin):
            begin -= 1

        end = boxes.end
        while not self._is_space(end):
            end += 1

        return Block(begin + 1, end)

    def _set_space(self, index: int):
        if index < self._remain_cell.begin or index >= self._remain_cell.end:
            return

        self._mark_cell(index, CellType.SPACE)
        for clue in self._clues_ex:
            clue.remove_candidates(index)

    def _get_known_boxes(self):
        known_boxes = []
        index = self._remain_cell.begin
        while index < self._remain_cell.end:
            if self._content[index] != CellType.BOX:
                index += 1
                continue

            begin = index
            while index != self._remain_cell.end and self._content[index] == CellType.BOX:
                index += 1

            block = Block(begin, index)
            clues = list(self._clues_ex)
            known_boxes.append(BoxBlockAndClues(block, clues))

        return known_boxes

    def _can_join(self, block1: Block, block2: Block, clue_value: int) -> bool:
        if block2.end - block1.begin > clue_value:
            return False

        return not any(self._is_space(i) for i in range(block1.end, block2.begin))

    def _handle_known_boxes(self, known_boxes) -> bool:
        updated = False
        index = 0
        while index < len(known_boxes):
            block, clues = known_boxes[index]
            candidate_clues = list(clues)
            clues.clear()
            boundary = self._find_boundary(block)

            for clue in candidate_clues:
                can_contain = clue.can_contain_box(block, boundary)
                if can_contain == Ternary.EXCLUSIVE:
                    clues.clear()
                    clues.append(clue)
                    clue.confirm_boxes(block)
                    updated = True
                    break
                elif can_contain == Ternary.YES:
                    clues.append(clue)

            if not clues:
                raise ParadoxError(f'boxes {block} cannot be matched to any clue')
            elif len(clues) > 1:
                clue_min = min(clue.value for clue in clues)
                clue_max = max(clue.value for clue in clues)

                if block.length < clue_min:
                    temp_clue = ClueExtra(math.nan, clue_min, boundary.begin, boundary.end)
                    temp_clue.confirm_boxes(block)
                    if temp_clue.boxes.length > block.length:
                        for i in temp_clue.boxes.iter():
                            self._mark_cell(i, CellType.BOX)

                        begin = temp_clue.boxes.begin
                        end = temp_clue.boxes.end

                        if index < len(known_boxes) - 1 and known_boxes[index + 1].block.begin <= end:
                            end = known_boxes[index + 1].block.end
                            del known_boxes[index + 1]

                        if index > 0 and known_boxes[index - 1].block.end >= begin:
                            begin = known_boxes[index - 1].block.begin
                            del known_boxes[index - 1]
                            index -= 1

                        block = Block(begin, end)
                        known_boxes[index] = BoxBlockAndClues(block, clues)
                        updated = True

                        if block.length > temp_clue.boxes.length:
                            continue

                if clue_max == clue_min == block.length and boundary.length > block.length:
                    self._set_space(block.begin - 1)
                    self._set_space(block.end)
                    updated = True

            index += 1

        # Push prev and next known boxes' clues.
        for index in range(len(known_boxes) - 1):
            block, clues = known_boxes[index]
            next_block, next_clues = known_boxes[index + 1]
            while next_clues and next_clues[0].index < clues[0].index:
                next_clues.pop(0)

            if next_clues and next_clues[0].index == clues[0].index:
                if not self._can_join(block, next_block, clues[0].value):
                    next_clues.pop(0)

            if not next_clues:
                raise ParadoxError(f'boxes {next_block} cannot be matched to any clue')

        for index in range(len(known_boxes) - 1, 1, -1):
            block, clues = known_boxes[index]
            prev_block, prev_clues = known_boxes[index - 1]
            while prev_clues and prev_clues[-1].index > clues[-1].index:
                prev_clues.pop()
                updated = True

            if prev_clues and prev_clues[-1].index == clues[-1].index:
                if not self._can_join(prev_block, block, clues[-1].value):
                    prev_clues.pop()
                    updated = True

            if not prev_clues:
                raise ParadoxError(f'boxes {prev_block} cannot be matched to any clue')

        index = 0
        while index < len(known_boxes):
            block, clues = known_boxes[index]
            if len(clues) == 1:
                clues[0].confirm_boxes(block)
                del known_boxes[index]
                updated = True
            else:
                index += 1

        return updated

    def _mark_boxes(self):
        for clue in self._clues_ex:
            if clue.boxes:
                for i in clue.boxes.iter():
                    self._mark_cell(i, CellType.BOX)

    def _mark_spaces(self):
        remain = BlockSection(self._remain_cell.begin, self._remain_cell.end)
        for clue in self._clues_ex:
            remain -= clue.candidates

        for i in remain.iter():
            self._mark_cell(i, CellType.SPACE)


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

    def solve_line(self, clues: typing.Tuple[int], content: typing.List[CellType], line: Line=None) -> typing.Set[int]:
        line_solver = LineSolver(clues, content)
        try:
            line_solver.solve()
        except ParadoxError as e:
            if line:
                raise ParadoxError(f'paradox in {line}: {e}') from e
            else:
                raise

        return line_solver.changes


def format_line(content: typing.List[CellType], col_fence=0) -> str:
    mapping = { CellType.BOX: '@', CellType.SPACE: '*' }
    line = []
    for col, value in enumerate(content):
        if col > 0 and col_fence > 0 and col % col_fence == 0:
            line.append('|')
        ch = mapping.get(value, '.')
        line.append(ch)

    return ''.join(line)


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
    subparsers = parser.add_subparsers(dest='mode', required=True, help='choose a mode')

    parser_g = subparsers.add_parser('gram', help='gram mode')
    parser_g.add_argument('puzzle_file', nargs='?',
                          help='a file contains the nanogram puzzle, see puzzles/*.txt for example'
                          ' (default: read from stdin)')

    parser_l = subparsers.add_parser('line', help='single line mode')
    parser_l.add_argument('length', type=int,
                         help='length of line')
    parser_l.add_argument('clues', type=int, nargs='+', metavar='clue',
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
    if args.mode == 'gram':
        puzzle = load_puzzle(args.puzzle_file)
        print(puzzle.row_clues)
        print(puzzle.col_clues)
        print(puzzle.board)
    elif args.mode == 'line':
        content = parse_line_content(args.content, args.length)
        solver.solve_line(args.clues, content)
        print(format_line(content, col_fence=5))


if __name__ == '__main__':
    main()
