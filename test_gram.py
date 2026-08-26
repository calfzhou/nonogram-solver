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
        solver.guess_enabled = True
        puzzle = solver.io.load_puzzle(gram_file_path)

        solver.pre_check(puzzle)
        board = solver.solve(puzzle)
        self.assertTrue(board.finished(), 'gram was not fully solved by guessing')
        solver.verify(puzzle, board)


if __name__ == '__main__':
    unittest.main()
