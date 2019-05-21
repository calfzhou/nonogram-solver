import glob
import unittest

import ddt

from nonogram_solver import *


def deducible_grams():
    return glob.glob('puzzles/*.txt', recursive=False)


@ddt.ddt
class TestCase(unittest.TestCase):
    @ddt.data(*deducible_grams())
    def test_deduce_gram(self, gram_file_path):
        puzzle = load_puzzle(gram_file_path)
        solver = NonogramSolver()

        solver.pre_check(puzzle)
        board = solver.solve(puzzle)
        self.assertTrue(board.finished())
        solver.verify(puzzle, board)


if __name__ == '__main__':
    unittest.main()
