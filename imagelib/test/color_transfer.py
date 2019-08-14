import unittest

import cv2
import numpy as np

from facelib import LandmarksProcessor
from imagelib import reinhard_color_transfer
from imagelib.color_transfer import _scale_array, lab_image_stats, linear_color_transfer
from samplelib import SampleLoader, SampleType


class MyTestCase(unittest.TestCase):
    def test_something(self):
        src_samples = SampleLoader.load(SampleType.FACE, './test_src', None)
        dst_samples = SampleLoader.load(SampleType.FACE, './test_dst', None)

        src_sample = src_samples[2]
        src_img = src_sample.load_bgr()
        src_mask = src_sample.load_fanseg_mask() or \
                   LandmarksProcessor.get_image_hull_mask(src_img.shape, src_sample.landmarks)

        grid = []
        for ct_sample in dst_samples:
            print(src_sample.filename, ct_sample.filename)
            ct_img = ct_sample.load_bgr()
            ct_mask = ct_sample.load_fanseg_mask() or \
                      LandmarksProcessor.get_image_hull_mask(ct_img.shape, ct_sample.landmarks)

            lct_img = linear_color_transfer(src_img, ct_img)
            rct_img = reinhard_color_transfer(src_img, ct_img)
            rct_img_clip = reinhard_color_transfer(src_img, ct_img, clip=True)
            rct_img_paper = reinhard_color_transfer(src_img, ct_img, preserve_paper=True)
            rct_img_clip_paper = reinhard_color_transfer(src_img, ct_img, clip=True, preserve_paper=True)

            masked_rct_img = reinhard_color_transfer(src_img, ct_img, source_mask=src_mask, target_mask=ct_mask)
            masked_rct_img_clip = reinhard_color_transfer(src_img, ct_img, clip=True, source_mask=src_mask, target_mask=ct_mask)
            masked_rct_img_paper = reinhard_color_transfer(src_img, ct_img, preserve_paper=True, source_mask=src_mask, target_mask=ct_mask)
            masked_rct_img_clip_paper = reinhard_color_transfer(src_img, ct_img, clip=True, preserve_paper=True, source_mask=src_mask, target_mask=ct_mask)
            results = np.concatenate((src_img, ct_img, lct_img, rct_img, rct_img_clip, rct_img_paper,
                                      rct_img_clip_paper, masked_rct_img, masked_rct_img_clip,
                                      masked_rct_img_paper, masked_rct_img_clip_paper), axis=1)
            grid.append(results)

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)
        # cv2.imshow('test output', cv2.hconcat([src_img, ct_img, lct_img, rct_img, rct_img_clip, rct_img_paper, rct_img_clip_paper], 7))

        cv2.imshow('test output', np.concatenate(grid, axis=0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.assertEqual(True, False)

    def test_reinhard_color_tranfer(self):
        src_samples = SampleLoader.load(SampleType.FACE, './test_src', None)
        dst_samples = SampleLoader.load(SampleType.FACE, './test_dst', None)

        clip = True
        preserve_paper = False
        source_mask = None
        target_mask = None

        cur_sample = src_samples[0]
        ct_sample = dst_samples[0]

        source = cur_sample.load_bgr()
        source_mask = cur_sample.load_fanseg_mask() or LandmarksProcessor.get_image_hull_mask(source.shape, cur_sample.landmarks)
        target = ct_sample.load_bgr()
        target_mask = ct_sample.load_fanseg_mask() or LandmarksProcessor.get_image_hull_mask(target.shape, ct_sample.landmarks)

        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

        # compute color statistics for the source and target images
        (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = lab_image_stats(source, mask=None)
        (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = lab_image_stats(target, mask=None)

        print(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc)
        print(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar)

        (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = lab_image_stats(source, mask=source_mask)
        (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = lab_image_stats(target, mask=target_mask)

        print(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc)
        print(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar)

        # subtract the means from the target image
        (l, a, b) = cv2.split(source)
        l -= lMeanSrc
        a -= aMeanSrc
        b -= bMeanSrc

        if preserve_paper:
            # scale by the standard deviations using paper proposed factor
            l = (lStdTar / lStdSrc) * l
            a = (aStdTar / aStdSrc) * a
            b = (bStdTar / bStdSrc) * b
        else:
            # scale by the standard deviations using reciprocal of paper proposed factor
            l = (lStdSrc / lStdTar) * l
            a = (aStdSrc / aStdTar) * a
            b = (bStdSrc / bStdTar) * b

        # add in the source mean
        l += lMeanTar
        a += aMeanTar
        b += bMeanTar

        # clip/scale the pixel intensities to [0, 255] if they fall
        # outside this range
        l = _scale_array(l, 0, 100, clip=clip)
        a = _scale_array(a, -127, 127, clip=clip)
        b = _scale_array(b, -127, 127, clip=clip)

        # merge the channels together and convert back to the RGB color
        transfer = cv2.merge([l, a, b])
        transfer = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
        np.clip(transfer, 0, 1, out=transfer)

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)
        cv2.imshow('test output', cur_sample.load_bgr())
        cv2.waitKey(0)
        cv2.imshow('test output', source_mask)
        cv2.waitKey(0)
        cv2.imshow('test output', ct_sample.load_bgr())
        cv2.waitKey(0)
        cv2.imshow('test output', target_mask)
        cv2.waitKey(0)
        cv2.imshow('test output', transfer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    unittest.main()
