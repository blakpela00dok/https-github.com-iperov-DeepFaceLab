import unittest

import cv2
import numpy as np

from mainscripts.Extractor import ExtractSubprocessor
from nnlib import nnlib
from facelib import LandmarksExtractor, S3FDExtractor


class LandmarkExtractorTest(unittest.TestCase):
    def test_extract(self):
        im = cv2.imread('../../imagelib/test/test_src/carrey/carrey.jpg')
        h, w, _ = im.shape

        device_config = nnlib.DeviceConfig(cpu_only=True)
        nnlib.import_all(device_config)
        landmark_extractor = LandmarksExtractor(nnlib.keras)
        s3fd_extractor = S3FDExtractor()

        rects = s3fd_extractor.extract(input_image=im, is_bgr=True)
        print('rects:', rects)
        l, t, r, b = rects[0]

        landmark_extractor.__enter__()
        # landmarks = landmark_extractor.extract(input_image=im, rects=rects, second_pass_extractor=None,
        #                                        is_bgr=True)
        s3fd_extractor.__enter__()
        landmarks = landmark_extractor.extract(input_image=im, rects=rects, second_pass_extractor=s3fd_extractor,
                                               is_bgr=True)[-1]
        s3fd_extractor.__exit__()
        landmark_extractor.__exit__()

        # print('landmarks', list(landmarks))

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)
        cv2.imshow('test output', im)
        cv2.waitKey(0)

        cv2.rectangle(im, (l, t), (r, b), (255, 255, 0))
        cv2.imshow('test output', im)
        cv2.waitKey(0)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.25

        def pt(arr=None, x=None, y=None):
            if x and y:
                return int(x), int(y)
            else:
                return int(arr[0]), int(arr[1])

        for i, m in enumerate(landmarks):
            print(i, m)
            cv2.circle(im, pt(m), 3, (0, 255, 0), thickness=-1)
            cv2.putText(im, str(i), pt(m), font_face, font_scale, (0, 255, 0), thickness=1)
        cv2.imshow('test output', im)
        cv2.waitKey(0)

        l_eyebrow = np.mean(landmarks[17:22, :], axis=0)
        r_eyebrow = np.mean(landmarks[22:27, :], axis=0)
        print(l_eyebrow, r_eyebrow)
        cv2.circle(im, pt(l_eyebrow), 5, (0, 0, 255))
        cv2.circle(im, pt(r_eyebrow), 5, (0, 0, 255))

        c_brow = np.mean([l_eyebrow, r_eyebrow], axis=0)
        brow_slope = (r_eyebrow[1] - l_eyebrow[1]) / (r_eyebrow[0] - l_eyebrow[0])
        l_brow_line = c_brow - np.array([1000, 1000 * brow_slope])
        r_brow_line = c_brow + np.array([1000, 1000 * brow_slope])
        cv2.line(im, pt(l_brow_line), pt(r_brow_line), (0, 0, 255), thickness=4)

        cv2.circle(im, pt(c_brow), 5, (0, 0, 255))
        nose = np.mean([landmarks[31], landmarks[35]], axis=0)
        cv2.circle(im, pt(nose), 5, (0, 0, 255))

        nose_brow_slope = (c_brow[1] - nose[1]) / (c_brow[0] - nose[0])
        t_nose_brow_line = c_brow - np.array([100, 100 * nose_brow_slope])
        b_nose_brow_line = c_brow + np.array([100, 100 * nose_brow_slope])
        cv2.line(im, pt(b_nose_brow_line), pt(t_nose_brow_line), (0, 0, 255), thickness=4)

        l_nose_line = nose - np.array([100, 100 * brow_slope])
        r_nose_line = nose + np.array([100, 100 * brow_slope])
        print(l_nose_line, r_nose_line)
        cv2.line(im, pt(l_nose_line), pt(r_nose_line), (0, 0, 255), thickness=1)

        c_forehead = c_brow - (nose - c_brow)
        cv2.circle(im, pt(c_forehead), 5, (0, 0, 255))
        l_forehead_line = c_forehead - np.array([100, 100 * brow_slope])
        r_forehead_line = c_forehead + np.array([100, 100 * brow_slope])
        cv2.line(im, pt(l_forehead_line), pt(r_forehead_line), (0, 0, 255), thickness=1)

        def mirrorUsingLine(pts, line_pt1, line_pt2):
            pass

        cv2.imshow('test output', im)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
