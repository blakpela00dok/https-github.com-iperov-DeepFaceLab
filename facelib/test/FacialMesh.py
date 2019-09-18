import random
import time
import unittest

import cv2
import numpy as np

from facelib.FacialMesh import get_mesh_landmarks
from nnlib import nnlib
from facelib import LandmarksExtractor, S3FDExtractor
from samplelib import SampleLoader, SampleType


class MyTestCase(unittest.TestCase):
    def test_something(self):
        t0 = time.time()
        source_image = cv2.imread('../../imagelib/test/test_src/carrey/carrey.jpg')
        print(time.time() - t0, 'loaded image')
        print('source_image type:', source_image.dtype)
        print('source_image shape:', source_image.shape)
        im = np.copy(source_image)

        device_config = nnlib.DeviceConfig(cpu_only=True)
        nnlib.import_all(device_config)
        landmark_extractor = LandmarksExtractor(nnlib.keras)
        s3fd_extractor = S3FDExtractor()

        rects = s3fd_extractor.extract(input_image=im, is_bgr=True)
        print('rects:', rects)
        bbox = rects[0]  # bounding box
        l, t, r, b = bbox

        print(time.time() - t0, 'got bbox')
        landmark_extractor.__enter__()
        s3fd_extractor.__enter__()

        landmarks = landmark_extractor.extract(input_image=im, rects=rects, second_pass_extractor=s3fd_extractor,
                                               is_bgr=True)[-1]
        s3fd_extractor.__exit__()
        landmark_extractor.__exit__()
        print(time.time() - t0, 'got landmarks')
        print('landmarks shape:', np.shape(landmarks))

        mesh_points, isomap, mask = get_mesh_landmarks(landmarks, im)
        print(time.time() - t0, 'got mesh')
        print('mesh_points:', np.shape(mesh_points))

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)

        # Draw the bounding box
        cv2.rectangle(im, (l, t), (r, b), (0, 0, 255), thickness=2)

        for i, pt in enumerate(mesh_points):
            cv2.circle(im, (int(pt[0]), int(pt[1])), 1, (255, 255, 255), thickness=-1)

        # Draw the landmarks
        for i, pt in enumerate(landmarks):
            cv2.circle(im, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), thickness=-1)

        cv2.imshow('test output', im)
        cv2.waitKey(0)

        cv2.imshow('test output', isomap.transpose([1, 0, 2]))
        cv2.waitKey(0)

        im = np.copy(source_image).astype(np.float32) / 255.0

        cv2.imshow('test output', mask)
        cv2.waitKey(0)

        cv2.imshow('test output', mask * im)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    def test_compare_hull_mask_with_mesh_mask(self):
        src_samples = SampleLoader.load(SampleType.FACE, '../../imagelib/test/test_dst', None)

        sample_grid = self.get_sample_grid(src_samples)
        display_grid = []
        for sample_row in sample_grid:
            display_row = []
            for sample in sample_row:
                src_img = sample.load_bgr()
                src_hull_mask = sample.load_image_hull_mask()
                src_mesh_mask = sample.load_image_mesh_mask()

                results = np.concatenate((src_img, src_hull_mask * src_img, src_mesh_mask * src_img), axis=1)
                display_row.append(results)
            display_grid.append(np.concatenate(display_row, axis=1))
        output_grid = np.concatenate(display_grid, axis=0)

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)
        cv2.imshow('test output', output_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def get_sample_grid(src_samples):
        pitch_yaw = np.array([[sample.pitch_yaw_roll[0], sample.pitch_yaw_roll[1]] for sample in src_samples])
        pitch_yaw = (pitch_yaw - np.mean(pitch_yaw, axis=0)) / np.std(pitch_yaw, axis=0)

        grid = [[[1, 1], [1, 0], [1, -1]],
                [[0, 1], [0, 0], [0, -1]],
                [[-1, 1], [-1, 0], [-1, -1]]]

        grid_samples = []
        for row in grid:
            row_samples = []
            for item in row:
                row_samples.append(src_samples[np.sum(np.square(np.abs(pitch_yaw - item)), 1).argmin()])
            grid_samples.append(row_samples)
        return grid_samples


if __name__ == '__main__':
    unittest.main()
