import unittest

import cv2
import numpy as np

from facelib.LandmarksProcessor import draw_landmarks
from samplelib import SampleLoader, SampleType


class LandmarksProcessorTests(unittest.TestCase):
    def test_algorithms(self):
        src_samples = SampleLoader.load(SampleType.FACE, '../../imagelib/test/test_dst', None)

        grid = []
        for src_sample in src_samples:
            src_img = src_sample.load_bgr()
            src_mask = src_sample.load_image_hull_mask()
            src_landmarks = src_sample.landmarks
            draw_landmarks(src_img, src_landmarks)
            results = np.concatenate((src_img, src_mask*src_img), axis=1)
            grid.append(results)

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)
        for g in grid:
            print(np.shape(g))
        cv2.imshow('test output', np.concatenate(grid, axis=0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_plot_landmarks_algorithms(self):
        src_samples = SampleLoader.load(SampleType.FACE, '../../imagelib/test/test_src', None)

        grid = []
        for src_sample in src_samples:
            src_img = src_sample.load_bgr()
            src_mask = src_sample.load_image_hull_mask()
            src_landmarks = src_sample.landmarks
            print('landmarks:', src_landmarks)
            for landmark in src_landmarks:
                landmark = np.array(landmark, dtype=np.int)
                cv2.circle(src_img, tuple(landmark), 3, (0,0,255))
            results = np.concatenate((src_img, src_mask*src_img), axis=1)
            grid.append(results)

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)
        for g in grid:
            print(np.shape(g))
        cv2.imshow('test output', np.concatenate(grid, axis=0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    unittest.main()
