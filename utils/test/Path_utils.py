import unittest

from utils.Path_utils import get_first_file_by_stem, get_image_paths


class PathUtilsTests(unittest.TestCase):
    def test_get_first_file_by_stem(self):
        first_file = get_first_file_by_stem('../..', 'README')
        self.assertEqual(first_file.stem, 'README')
        self.assertEqual(first_file.suffix, '.md')
        self.assertEqual(first_file.name, 'README.md')

    def test_get_image_paths(self):
        image_paths = get_image_paths('..', ['.py'])
        print('Image paths for ".." (.py extension):', image_paths)


if __name__ == '__main__':
    unittest.main()
