import random
import time
import unittest

import cv2
import numpy as np

# from facelib.FacialMesh import get_mesh_landmarks
from facelib.LandmarksProcessor import get_image_hull_mask, calc_image_size_for_unscaled, \
    get_transform_mat
from nnlib import nnlib
from facelib import LandmarksExtractor, S3FDExtractor, FaceType
from samplelib import SampleLoader, SampleType


class MyTestCase(unittest.TestCase):
    def test_something(self):
        t0 = time.time()
        source_image = cv2.imread('test_image/carrey.jpg')
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

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)

        # Draw the bounding box
        cv2.rectangle(im, (l, t), (r, b), (0, 0, 255), thickness=2)

        # Draw the landmarks
        for i, pt in enumerate(landmarks):
            cv2.circle(im, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), thickness=-1)

        cv2.imshow('test output', im)
        cv2.waitKey(0)

        face_type = FaceType.FULL
        size = calc_image_size_for_unscaled(landmarks, face_type)
        # size = 480
        mat = get_transform_mat(landmarks, size, face_type)
        face_image = cv2.warpAffine(im, mat, (size, size), cv2.INTER_LANCZOS4)

        cv2.imshow('test output', face_image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
