import glob
import io
import unittest
from contextlib import redirect_stdout

import ddt

from nonogram_solver import *


class RejectFirstGuessSolver(NonogramSolver):
    def __init__(self):
        super().__init__()
        self.probe_calls = 0

    def _find_contradiction_deduction(self, puzzle, board):
        self.probe_calls += 1
        if self.probe_calls == 2:
            raise ParadoxError('reject first guess')
        return None


def deducible_grams():
    return glob.glob('puzzles/*.txt', recursive=False)


def need_guess_grams():
    return glob.glob('puzzles/need-guess/*.txt', recursive=False)


@ddt.ddt
class TestCase(unittest.TestCase):
    @ddt.data(*deducible_grams())
    def test_deduce_gram(self, gram_file_path):
        solver = NonogramSolver()
        puzzle = solver.io.load_puzzle(gram_file_path)

        solver.pre_check(puzzle)
        board = solver.solve(puzzle)
        self.assertTrue(board.finished(), 'gram was not fully solved')
        solver.verify(puzzle, board)

    @ddt.data(*need_guess_grams())
    def test_probe_gram(self, gram_file_path):
        solver = NonogramSolver()
        puzzle = solver.io.load_puzzle(gram_file_path)

        solver.pre_check(puzzle)

        # Basic line deduction still stalls on this puzzle.
        solver.probe_enabled = False
        solver.guess_enabled = False
        deduce_board = solver.solve(puzzle)
        self.assertFalse(
            deduce_board.finished(),
            f'puzzle is now solvable by line deduction alone; '
            f'move {gram_file_path} out of puzzles/need-guess/',
        )

        # Contradiction probing must solve it without committing a guess.
        solver.probe_enabled = True
        board = solver.solve(puzzle)
        self.assertTrue(board.finished(), 'gram was not fully solved by contradiction deduction')
        solver.verify(puzzle, board)

    def test_probe_paradox_backtracks_guess(self):
        puzzle = NonogramPuzzle(((1,),) * 3, ((1,),) * 3, None)
        solver = RejectFirstGuessSolver()
        solver.guess_enabled = True
        solver.probe_enabled = True

        board = solver.solve(puzzle)

        self.assertGreaterEqual(solver.probe_calls, 2)
        self.assertTrue(board.finished(), 'gram was not fully solved after probe backtracking')
        solver.verify(puzzle, board)

    def test_show_probe(self):
        puzzle = NonogramPuzzle(((1,),), ((1,),), None)
        solver = NonogramSolver()
        solver.probing_visible = True
        output = io.StringIO()

        with redirect_stdout(output):
            forced_cell = solver._find_contradiction_deduction(puzzle, Board(1, 1))

        self.assertEqual((Coord(0, 0), CellType.BOX), forced_cell)
        self.assertEqual(
            [
                '[Probe] assume cell [1, 1] is BOX',
                '[Probe] no contradiction found',
                '[Probe] assume cell [1, 1] is SPACE',
                '[Probe paradox] paradox in ROW 1: boxes (0,) do not match clues (1,)',
                '[Probe deduction] cell [1, 1] must be BOX',
                '',
            ],
            output.getvalue().splitlines(),
        )


if __name__ == '__main__':
    unittest.main()
