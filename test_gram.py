import glob
import unittest

import ddt

from nonogram_solver import *


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
    def test_guess_gram(self, gram_file_path):
        solver = NonogramSolver()
        puzzle = solver.io.load_puzzle(gram_file_path)

        solver.pre_check(puzzle)

        # 1) deduction only must NOT finish -- the LOUD ALERT fires here if
        #    a solver improvement ever makes this puzzle deducible.
        solver.guess_enabled = False
        deduce_board = solver.solve(puzzle)
        self.assertFalse(
            deduce_board.finished(),
            f'needs-guessing puzzle is now solveable by deduction only; '
            f'move {gram_file_path} out of puzzles/need-guess/',
        )

        # 2) deduction stalled -> guessing on the SAME solver + SAME puzzle.
        solver.guess_enabled = True
        board = solver.solve(puzzle)
        self.assertTrue(board.finished(), 'gram was not fully solved by guessing')
        solver.verify(puzzle, board)

    def test_guess_rejects_invalid_finished_board(self):
        puzzle = NonogramPuzzle(
            ((0,), (2,), (1, 1)),
            ((1,), (1,), (1,), (1,)),
            None,
        )
        solver = NonogramSolver()
        solver.guess_enabled = True

        board = solver.solve(puzzle)

        self.assertTrue(board.finished(), 'gram was not fully solved by guessing')
        solver.verify(puzzle, board)


if __name__ == '__main__':
    unittest.main()
