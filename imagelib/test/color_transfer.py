import unittest

import cv2
import numpy as np

from facelib import LandmarksProcessor
from imagelib import reinhard_color_transfer
from imagelib.color_transfer import _scale_array, lab_image_stats, linear_color_transfer
from interact.interact import InteractDesktop
from samplelib import SampleLoader, SampleType


class ColorTranfer(unittest.TestCase):
    def test_algorithms(self):
        src_samples = SampleLoader.load(SampleType.FACE, './test_src', None)
        dst_samples = SampleLoader.load(SampleType.FACE, './test_dst', None)

        src_sample = src_samples[2]
        src_img = src_sample.load_bgr()
        src_mask = src_sample.load_mask()

        # Toggle to see masks
        show_masks = False

        grid = []
        for ct_sample in dst_samples:
            print(src_sample.filename, ct_sample.filename)
            ct_img = ct_sample.load_bgr()
            ct_mask = ct_sample.load_mask()

            lct_img = linear_color_transfer(src_img, ct_img)
            rct_img = reinhard_color_transfer(src_img, ct_img)
            rct_img_clip = reinhard_color_transfer(src_img, ct_img, clip=True)
            rct_img_paper = reinhard_color_transfer(src_img, ct_img, preserve_paper=True)
            rct_img_paper_clip = reinhard_color_transfer(src_img, ct_img, clip=True, preserve_paper=True)

            masked_rct_img = reinhard_color_transfer(src_img, ct_img, source_mask=src_mask, target_mask=ct_mask)
            masked_rct_img_clip = reinhard_color_transfer(src_img, ct_img, clip=True, source_mask=src_mask, target_mask=ct_mask)
            masked_rct_img_paper = reinhard_color_transfer(src_img, ct_img, preserve_paper=True, source_mask=src_mask, target_mask=ct_mask)
            masked_rct_img_paper_clip = reinhard_color_transfer(src_img, ct_img, clip=True, preserve_paper=True, source_mask=src_mask, target_mask=ct_mask)

            results = [lct_img, rct_img, rct_img_clip, rct_img_paper, rct_img_paper_clip,
                       masked_rct_img, masked_rct_img_clip, masked_rct_img_paper, masked_rct_img_paper_clip]

            if show_masks:
                results = [src_mask * im for im in results]
                src_img *= src_mask
                ct_img *= ct_mask

            results = np.concatenate((src_img, ct_img, *results), axis=1)
            grid.append(results)

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)
        cv2.imshow('test output', np.concatenate(grid, axis=0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
