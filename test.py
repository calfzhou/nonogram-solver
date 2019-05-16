import unittest

import ddt

from nonogram_solver import *


@ddt.ddt
class TestCase(unittest.TestCase):
    @ddt.file_data('test-data/line-cases.yaml')
    def test_line_solver(self, **case_info):
        length = case_info['length']
        clues = case_info['clues']
        marked = case_info.get('marked', '')
        line = parse_line_content(marked, length)
        # print(length, clues, line)
        result = case_info.get('result')
        solver = NonogramSolver()

        try:
            solver.solve_line(clues, line)
            self.assertIsNotNone(result)
            expected = parse_line_content(result, length)
            self.assertSequenceEqual(format_line(line, 5), format_line(expected, 5))
        except ParadoxError:
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
