import unittest

import ddt

from nonogram_solver import *


@ddt.ddt
class TestCase(unittest.TestCase):
    @ddt.file_data('test-data/line-cases.yaml')
    def test_line_solver(self, **case_info):
        length = case_info['length']
        clues = case_info['clues']
        origin = case_info.get('origin', '')
        result = case_info.get('result')

        solver = NonogramSolver()
        solver.io.line_fence = 5
        line = solver.io.parse_line(origin, length)

        try:
            solver.solve_line(clues, line)
        except ParadoxError:
            self.assertIsNone(result)
        else:
            self.assertIsNotNone(result)
            expected = solver.io.parse_line(result, length)
            self.assertSequenceEqual(solver.io.format_line(line), solver.io.format_line(expected))


if __name__ == '__main__':
    unittest.main()
