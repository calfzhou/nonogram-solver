import unittest

import ddt

from nonogram_solver import *


@ddt.ddt
class TestCase(unittest.TestCase):
    @ddt.file_data('test-data/line-cases.yaml')
    def test_line_solver(self, **case_info):
        length = case_info['length']
        hints = case_info['hints']
        marked = case_info.get('marked', '')
        line = parse_line_content(marked, length)
        # print(length, hints, line)
        result = case_info.get('result')
        solver = NonogramSolver()

        try:
            solver.solve_line(hints, line)
            self.assertIsNotNone(result)
            expected = parse_line_content(result, length)
            self.assertSequenceEqual(None, expected)
        except ParadoxError:
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
