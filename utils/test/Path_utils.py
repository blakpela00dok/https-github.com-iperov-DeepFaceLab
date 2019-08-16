import unittest

from utils.Path_utils import get_first_file_by_stem


class PathUtilsTests(unittest.TestCase):
    def test_get_first_file_by_stem(self):
        first_file = get_first_file_by_stem('../..', 'README')
        self.assertEqual(first_file.stem, 'README')
        self.assertEqual(first_file.suffix, '.md')
        self.assertEqual(first_file.name, 'README.md')


if __name__ == '__main__':
    unittest.main()
