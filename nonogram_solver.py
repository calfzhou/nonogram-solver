#!/usr/bin/env python3

import argparse
import distutils.util
import collections
import copy
import enum
import itertools
import math
import sys
import time
import typing

__version__ = '1.0.0'


class CellType(enum.Enum):
    BOX = enum.auto()
    SPACE = enum.auto()

    def __str__(self):
        return self.name


class LineKind(enum.Enum):
    ROW = enum.auto()
    COL = enum.auto()

    def orthogonal(self) -> 'LineKind':
        if self is self.ROW:
            return self.COL
        elif self is self.COL:
            return self.ROW
        else:
            raise ValueError(f'{self} has no orthogonal line')

    def __str__(self):
        return self.name

class Coord(typing.NamedTuple):
    row: int
    col: int

    def __str__(self):
        return f'[{self.row + 1}, {self.col + 1}]'


class Line(typing.NamedTuple):
    kind: LineKind
    n: int

    def get_coord(self, index: int) -> Coord:
        if self.kind == LineKind.ROW:
            return Coord(self.n, index)
        elif self.kind == LineKind.COL:
            return Coord(index, self.n)

    def __str__(self):
        return f'{self.kind} {self.n + 1}'


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

    def get_line_content(self, line: Line) -> typing.Tuple[CellType]:
        length = self._width if (line.kind == LineKind.ROW) else self._height
        return [self[line.get_coord(i)] for i in range(length)]

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

    def get_line_clues(self, line: Line) -> typing.Tuple[int]:
        if line.kind == LineKind.ROW:
            return self.row_clues[line.n]
        elif line.kind == LineKind.COL:
            return self.col_clues[line.n]


class GuessData(typing.NamedTuple):
    coord: Coord
    board: Board


class ParadoxError(Exception):
    pass


class FailedError(Exception):
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
        return f'[{self.begin + 1}-{self.end}]'

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
            raise ParadoxError(f'clue #{self._index + 1} (value {self._value}) has no box candidates')
        return self._candidates

    @property
    def boxes(self) -> Block:
        return self._boxes

    def get_prev(self) -> 'ClueExtra':
        return self._prev

    def get_next(self) -> 'ClueExtra':
        return self._next

    def finished(self) -> bool:
        return self._boxes and self._boxes.length == self._value

    def __repr__(self):
        return f'#{self._index + 1} ({self._value}) {self._boxes}'

    @classmethod
    def chain(cls, clues: typing.Iterable):
        prev: ClueExtra = None
        for curr in clues:
            if prev:
                prev._set_next(curr)
                curr._set_prev(prev)
            prev = curr

    def _off_chain(self):
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
        if self.finished():
            self._off_chain()

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

        for index in range(len(known_boxes) - 1, 0, -1):
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

        # Check confirmed boxes and special space, push other clues' candidates range.
        index = 0
        while index < len(known_boxes):
            block, clues = known_boxes[index]
            if len(clues) == 1:
                clues[0].confirm_boxes(block)
                del known_boxes[index]
                updated = True
                continue

            # Push prev and next clues' candidates range.
            prev_clue = clues[0].get_prev()
            if prev_clue and prev_clue.candidates.end >= block.begin:
                prev_clue.remove_tail_candidates(block.begin - 1)
                updated = True

            next_clue = clues[-1].get_next()
            if next_clue and next_clue.candidates.begin <= block.end:
                next_clue.remove_head_candidates(block.end + 1)
                updated = True

            # Special space.
            if block.begin == clues[-1].candidates.begin and block.length < clues[-1].value:
                if all(block.length == c.value for c in clues[:-1]) and not self._is_space(block.begin - 1):
                    self._set_space(block.begin - 1)
                    updated = True
            elif block.end == clues[0].candidates.end and block.length < clues[0].value:
                if all(block.length == c.value for c in clues[1:]) and not self._is_space(block.end):
                    self._set_space(block.end)
                    updated = True

            index += 1

        # Check force splitting.
        for index in range(len(known_boxes) - 1):
            block, clues = known_boxes[index]
            next_block, next_clues = known_boxes[index + 1]
            if block.end == next_block.begin - 1 and not self._is_space(block.end):
                separate = False
                if clues[-1].index < next_clues[0].index:
                    separate = True
                else:
                    shared_clues = set(clues)
                    shared_clues.intersection_update(next_clues)
                    merged = Block(block.begin, next_block.end)
                    can_merge = lambda c: merged.length <= c.value and merged in c.candidates
                    separate = not any(can_merge(c) for c in shared_clues)

                if separate:
                    self._set_space(block.end)
                    updated = True

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


class NonogramIO:
    class SymbolColl(typing.NamedTuple):
        box: str
        space: str
        unknown: str
        col_fence: str
        row_fence: str
        cross_fence: str

    def __init__(self):
        self.line_fence = 0
        self.row_fence = 0
        self.col_fence = 0
        self.full_width_enabled = False

        self.symbols = self.SymbolColl('@', '*', '.', '|', '-', '+')
        self.full_width_symbols = self.SymbolColl('䨻', 'ｘ', '、', '｜', '－', '＋')
        self.box_symbols = { 'o', self.symbols.box, self.full_width_symbols.box }
        self.space_symbols = { 'x', self.symbols.space, self.full_width_symbols.space }

    def format_line(self, content: typing.List[CellType], fence=None) -> str:
        fence = self.line_fence if fence is None else fence
        parts = []
        for col, value in enumerate(content):
            if self.line_fence > 0 and col > 0 and col % self.line_fence == 0:
                parts.append(self.symbols.col_fence)

            if value == CellType.BOX:
                parts.append(self.symbols.box)
            elif value == CellType.SPACE:
                parts.append(self.symbols.space)
            else:
                parts.append(self.symbols.unknown)

        return ''.join(parts)

    def format_board(self, board: Board, row_fence=None, col_fence=None, full_width=None, highlights=None) -> str:
        row_fence = self.row_fence if row_fence is None else row_fence
        col_fence = self.col_fence if col_fence is None else col_fence
        full_width = self.full_width_enabled if full_width is None else full_width
        symbols = self.full_width_symbols if full_width else self.symbols
        highlights = highlights or set()

        csi = '\x1b'
        c_highlight = csi + '[34m'
        c_reset = csi + '[0m'

        fence_parts = []
        for col in range(board.width):
            if col_fence > 0 and col > 0 and col % col_fence == 0:
                fence_parts.append(symbols.cross_fence)
            fence_parts.append(symbols.row_fence)

        fence_line = ''.join(fence_parts)

        lines = []
        for row in range(board.height):
            if row_fence > 0 and row > 0 and row % row_fence == 0:
                lines.append(fence_line)

            parts = []
            for col in range(board.width):
                if col_fence > 0 and col > 0 and col % col_fence == 0:
                    parts.append(symbols.col_fence)

                coord = Coord(row, col)
                if coord in highlights:
                    parts.append(c_highlight)

                value = board[coord]
                if value == CellType.BOX:
                    parts.append(symbols.box)
                elif value == CellType.SPACE:
                    parts.append(symbols.space)
                else:
                    parts.append(symbols.unknown)

                if coord in highlights:
                    parts.append(c_reset)

            lines.append(''.join(parts))

        return '\n'.join(lines)

    def parse_line(self, text: str, length: int) -> typing.List[CellType]:
        content = []
        for c in text:
            c = c.lower()
            if c in self.box_symbols:
                content.append(CellType.BOX)
            elif c in self.space_symbols:
                content.append(CellType.SPACE)
            elif c in { self.symbols.col_fence, self.full_width_symbols.col_fence }:
                continue
            else:
                content.append(None)

        if len(content) < length:
            content.extend(itertools.repeat(None, length - len(content)))
        elif len(content) > length:
            raise ValueError(f'line `{text}` is longer than given length {length}')

        return content

    def load_puzzle(self, file_path) -> NonogramPuzzle:
        row_clues = []
        col_clues = []
        board: Board = None
        with (open(file_path) if file_path else sys.stdin) as f:
            section = 0
            row = 0
            for line in f:
                line = line.rstrip('\r\n')
                if line.startswith('#'):
                    continue

                if section == 0:
                    if line == '':
                        section += 1
                    elif line[0] == self.symbols.row_fence:
                        continue
                    else:
                        row_clues.append(tuple(int(x) for x in line.split()))
                elif section == 1:
                    if line == '':
                        section += 1
                    elif line[0] == self.symbols.row_fence:
                        continue
                    else:
                        col_clues.append(tuple(int(x) for x in line.split()))
                elif section == 2:
                    if line == '':
                        break
                    elif line[0] in { self.symbols.row_fence, self.full_width_symbols.row_fence }:
                        continue
                    else:
                        board = board or Board(len(row_clues), len(col_clues))
                        row_content = self.parse_line(line, board.width)
                        for col, value in enumerate(row_content):
                            if value:
                                board[Coord(row, col)] = value

                        row += 1
                else:
                    break

        return NonogramPuzzle(tuple(row_clues), tuple(col_clues), board)


class NonogramSolver:
    def __init__(self):
        self.guess_enabled = False
        self.line_deduce_visible = False
        self.deduce_board_visible = False
        self.deduce_board_pause = 0
        self.guessing_visible = False

        self.io = NonogramIO()

    def solve(self, puzzle: NonogramPuzzle) -> Board:
        board = puzzle.board or Board(puzzle.height, puzzle.width)

        lines = collections.OrderedDict()
        lines.update((Line(LineKind.ROW, i), None) for i in range(board.height))
        lines.update((Line(LineKind.COL, i), None) for i in range(board.width))

        guesses: typing.List[GuessData] = []
        guessing = False

        while not board.finished():
            if not lines:
                if not self.guess_enabled:
                    break

                if not guessing:
                    board = copy.deepcopy(board)
                    guessing = True
                    if self.guessing_visible:
                        print('Deduce finished but not solved the puzzle, try guessing. The deduce result is:')
                        print(self.io.format_board(board))
                        print()

                # guess
                coord = self._choose_cell(board)
                guesses.append(GuessData(coord, copy.deepcopy(board)))
                board[coord] = CellType.BOX
                lines[Line(LineKind.ROW, coord.row)] = None
                lines[Line(LineKind.COL, coord.col)] = None
                if self.guessing_visible:
                    indent = '  ' * (len(guesses) - 1)
                    print(f'{indent}[Guess {len(guesses)}] assume cell {coord} is BOX')

            try:
                while lines:
                    line, _ = lines.popitem(last=False)
                    clues = puzzle.get_line_clues(line)
                    content = board.get_line_content(line)
                    origin = self.io.format_line(content)
                    changes = self.solve_line(clues, content, line)
                    if changes:
                        if self.line_deduce_visible:
                            print(f'solving {line}: {clues}')
                            print(f'origin: {origin}')
                            print(f'result: {self.io.format_line(content)}')
                            print()
                        orthogonal = line.kind.orthogonal()
                        for i in sorted(changes):
                            value = content[i]
                            assert value is not None, f'{line} cell {i + 1} set to None is meaningless'
                            coord = line.get_coord(i)
                            assert board[coord] is None, f'{line} cell {i + 1} is already confirmed'
                            if board[coord] is None and value is not None:
                                board[coord] = value
                                lines[Line(orthogonal, i)] = None
                        if self.deduce_board_visible:
                            highlights = set(line.get_coord(i) for i in changes)
                            print(self.io.format_board(board, highlights=highlights))
                            print()
                            time.sleep(self.deduce_board_pause)
            except ParadoxError as e:
                if guesses:
                    guess: GuessData = guesses.pop()
                    board = guess.board
                    board[guess.coord] = CellType.SPACE
                    lines[Line(LineKind.ROW, coord.row)] = None
                    lines[Line(LineKind.COL, coord.col)] = None
                    if self.guessing_visible:
                        indent = '  ' * (len(guesses) + 1)
                        print(f'{indent}[Paradox] {e}; so cell {guess.coord} should be SPACE')
                else:
                    print(self.io.format_board(board))
                    raise

        if guessing and self.guessing_visible and board.finished():
            indent = '  ' * len(guesses)
            print(f'{indent}[Success] find a valid solution')
            print()

        return board

    def _choose_cell(self, board: Board):
        for row in range(board.height):
            for col in range(board.width):
                coord = Coord(row, col)
                if board[coord] is None:
                    return coord

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

    def pre_check(self, puzzle: NonogramPuzzle):
        row_clue_sum = self._check_line_clue_sum(puzzle.row_clues, len(puzzle.col_clues), LineKind.ROW)
        col_clue_sum = self._check_line_clue_sum(puzzle.col_clues, len(puzzle.row_clues), LineKind.COL)
        if row_clue_sum != col_clue_sum:
            raise FailedError(f'total row clues sum ({row_clue_sum}) not equal to col clues sum ({col_clue_sum})')

    def _check_line_clue_sum(self, all_clues, length, line_kind):
        total = 0
        for i, clues in enumerate(all_clues):
            s = sum(clues)
            total += s
            min_len = s + len(clues) - 1
            if min_len > length:
                raise FailedError(f'{Line(line_kind, i)} clues {clues} cannot fit in {length} cells (at least {min_len})')

        return total

    def verify(self, puzzle: NonogramPuzzle, board: Board):
        row_boxes = [[0] for _ in range(board.height)]
        col_boxes = [[0] for _ in range(board.width)]
        # row_boxes = [None] * board.height
        # col_boxes = [None] * board.width
        for row in range(board.height):
            for col in range(board.width):
                value = board[Coord(row, col)]
                if value == CellType.BOX:
                    row_boxes[row][-1] += 1
                    col_boxes[col][-1] += 1
                elif value == CellType.SPACE:
                    if row_boxes[row][-1] > 0:
                        row_boxes[row].append(0)
                    if col_boxes[col][-1] > 0:
                        col_boxes[col].append(0)
                else:
                    raise FailedError(f'cell ({row + 1}, {col + 1}) is not marked')

        self._compare_boxes_with_clues(row_boxes, puzzle.row_clues, LineKind.ROW)
        self._compare_boxes_with_clues(col_boxes, puzzle.col_clues, LineKind.COL)

    def _compare_boxes_with_clues(self, all_boxes, all_clues, line_kind):
        assert len(all_boxes) == len(all_clues), f'the number of {line_kind}s do not match'
        for i in range(len(all_boxes)):
            boxes = tuple(filter(None, all_boxes[i])) or (0,)
            clues = all_clues[i]
            if boxes != clues:
                raise FailedError(f'{Line(line_kind, i)} boxes {boxes} not match with clues {clues}')


def create_arg_parser() -> argparse.ArgumentParser:
    strtobool = lambda s: bool(distutils.util.strtobool(s))
    int_pair = lambda s: tuple(int(x) for x in s.split(',', 1))

    parser = argparse.ArgumentParser(description='Nonograms Puzzle Solver', allow_abbrev=False)
    parser.add_argument('--version', action='version', version=f'nonograms_solver {__version__}')
    subparsers = parser.add_subparsers(dest='mode', required=True, help='choose a mode')

    parser_g = subparsers.add_parser('gram', help='gram mode')
    parser_g.add_argument('puzzle_file', nargs='?',
                          help='a file contains the nanogram puzzle, see puzzles/*.txt for example'
                          ' (default: read from stdin)')
    parser_g.add_argument('--guess', type=strtobool,
                          nargs='?', const=True, default=False, choices=[True, False],
                          help='whether enable guess when puzzle cannot be solved by deducing (default: false)')
    parser_g.add_argument('--show-progress', type=strtobool,
                          nargs='?', const=True, default=False, choices=[True, False],
                          help='whether print board after each deducing step (highlight changes) (default: false)')
    parser_g.add_argument('--progress-pause', type=float, default=0.2,
                          help='pause some time (in seconds) between each progress board view (default: 0.2)')
    parser_g.add_argument('--show-deduce', type=strtobool,
                          nargs='?', const=True, default=False, choices=[True, False],
                          help='whether print every line deducing result (default: false)')
    parser_g.add_argument('--show-guess', type=strtobool,
                          nargs='?', const=True, default=False, choices=[True, False],
                          help='whether print every guessing step (default: false)')
    parser_g.add_argument('--grid', type=int_pair, nargs='?', default=(0, 0), const=(5, 5), metavar='WIDTH[,HEIGHT]',
                          help='show major grid line when printing gram with the given size (default: 5,5)')
    parser_g.add_argument('--line-fence', type=int, default=5,
                          help='if greater than 0, print fence when printing single line (default: 5)')
    parser_g.add_argument('--full-width', type=strtobool,
                          nargs='?', const=True, default=True, choices=[True, False],
                          help='whether use full width char when print gram (default: true)')

    parser_l = subparsers.add_parser('line', help='single line mode')
    parser_l.add_argument('length', type=int,
                         help='length of line')
    parser_l.add_argument('clues', type=int, nargs='+', metavar='clue',
                          help='clue numbers')
    parser_l.add_argument('--content', default='',
                         help='content of the line, `o` or `@` for box, `x` or `*` for space,'
                         ' `|` for border (optional), other character for unknown (case insensitive)')
    parser_l.add_argument('--line-fence', type=int, default=5,
                          help='if greater than 0, print fence when printing single line (default: 5)')

    return parser


def create_solver(args) -> NonogramSolver:
    solver = NonogramSolver()
    solver.io.line_fence = args.line_fence
    if args.mode == 'gram':
        solver.guess_enabled = args.guess
        solver.deduce_board_visible = args.show_progress
        solver.deduce_board_pause = args.progress_pause
        solver.line_deduce_visible = args.show_deduce
        solver.guessing_visible = args.show_guess
        solver.io.col_fence = args.grid[0]
        solver.io.row_fence = args.grid[-1]
        solver.io.full_width_enabled = args.full_width

    return solver


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    # print(args)

    solver = create_solver(args)
    if args.mode == 'gram':
        puzzle = solver.io.load_puzzle(args.puzzle_file)
        solver.pre_check(puzzle)
        board = solver.solve(puzzle)
        print(solver.io.format_board(board))
        if board.finished():
            solver.verify(puzzle, board)
        else:
            print()
            print('NOT Solved!!!')
    elif args.mode == 'line':
        content = solver.io.parse_line(args.content, args.length)
        origin = solver.io.format_line(content)
        solver.solve_line(args.clues, content)
        print(f'solving line: {args.clues}')
        print(f'origin: {origin}')
        print(f'result: {solver.io.format_line(content)}')
        print()


if __name__ == '__main__':
    main()
