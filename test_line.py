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

        if 'fast_result' in case_info:
            fast_result = case_info['fast_result']
            fast_line = solver.io.parse_line(origin, length)
            try:
                solver.solve_line(clues, fast_line, exact=False)
            except ParadoxError:
                self.assertIsNone(fast_result, 'unexpected fast-stage paradox occurs')
            else:
                self.assertIsNotNone(fast_result, 'fast stage did not find paradox')
                expected = solver.io.parse_line(fast_result, length)
                self.assertSequenceEqual(
                    solver.io.format_line(fast_line),
                    solver.io.format_line(expected),
                    'not correctly solved by fast stage',
                )

        line = solver.io.parse_line(origin, length)

        try:
            solver.solve_line(clues, line)
        except ParadoxError:
            self.assertIsNone(result, 'unexpected paradox occurs')
        else:
            self.assertIsNotNone(result, 'did not find paradox')
            expected = solver.io.parse_line(result, length)
            self.assertSequenceEqual(
                solver.io.format_line(line), solver.io.format_line(expected), 'not correctly solved')


if __name__ == '__main__':
    unittest.main()
