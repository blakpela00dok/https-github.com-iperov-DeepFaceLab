import numpy as np
import cv2


def reinhard_color_transfer(target, source, clip=False, preserve_paper=False, target_mask=None, source_mask=None):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.

    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.

    Title: "Super fast color transfer between images"
    Author: Adrian Rosebrock
    Date: June 30. 2014
    Url: https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/

    Parameters:
    -------
    source: NumPy array
        OpenCV image (w, h, 3) in BGR color space (float32) 0-1
    target: NumPy array (float32)
        OpenCV image (w, h, 3) in BGR color space (float32), 0-1
    clip: Should components of L*a*b* image be scaled by np.clip before
        converting back to BGR color space?
        If False then components will be min-max scaled appropriately.
        Clipping will keep target image brightness truer to the input.30
        Scaling will adjust image brightness to avoid washed out portions
        in the resulting color transfer that can be caused by clipping.
    preserve_paper: Should color transfer strictly follow methodology
        layed out in original paper? The method does not always produce
        aesthetically pleasing results.
        If False then L*a*b* components will scaled using the reciprocal of
        the scaling factor proposed in the paper.  This method seems to produce
        more consistently aesthetically pleasing results

    Returns:
    -------
    transfer: NumPy array
        OpenCV image (w, h, 3) NumPy array (float32)
    """

    np.clip(source, 0, 1, out=source)
    np.clip(target, 0, 1, out=target)

    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source.astype(np.float32), cv2.COLOR_BGR2LAB)
    target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2LAB)

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = lab_image_stats(source, mask=source_mask)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = lab_image_stats(target, mask=target_mask)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

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
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall
    # outside this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)

    # merge the channels together and convert back to the RGB color
    # space
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
    np.clip(transfer, 0, 1, out=transfer)

    # return the color transferred image
    return transfer


def linear_color_transfer(target_img, source_img, mode='pca', eps=1e-5):
    """
    Matches the colour distribution of the target image to that of the source image
    using a linear transform.
    Images are expected to be of form (w,h,c) and float in [0,1].
    Modes are chol, pca or sym for different choices of basis.

    Title: "NeuralImageSynthesis / ExampleNotebooks / ScaleControl.ipynb"
    Author: Leon Gatys
    Date: December 14, 2016
    Url: https://github.com/leongatys/NeuralImageSynthesis/blob/master/ExampleNotebooks/ScaleControl.ipynb
    """
    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2, 0, 1).reshape(3, -1)
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
    mu_s = source_img.mean(0).mean(0)
    s = source_img - mu_s
    s = s.transpose(2, 0, 1).reshape(3, -1)
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        ts = chol_s.dot(np.linalg.pinv(chol_t)).dot(t)
    if mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        ts = Qs.dot(np.linalg.pinv(Qt)).dot(t)
    if mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.pinv(Qt)).dot(t)
    matched_img = ts.reshape(*target_img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
    matched_img += mu_s
    matched_img[matched_img > 1] = 1
    matched_img[matched_img < 0] = 0
    return matched_img


def lab_image_stats(image, mask=None):
    # compute the mean and standard deviation of each channel
    if mask is not None:
        mask = np.squeeze(mask)
        image = image[mask == 1]

    (l, a, b) = cv2.split(image)

    l_mean, l_std = np.mean(l), np.std(l)
    a_mean, a_std = np.mean(l), np.std(l)
    b_mean, b_std = np.mean(l), np.std(l)

    # return the color statistics
    return l_mean, l_std, a_mean, a_std, b_mean, b_std


def _scale_array(arr, clip=True):
    if clip:
        return np.clip(arr, 0, 255)

    mn = arr.min()
    mx = arr.max()
    scale_range = (max([mn, 0]), min([mx, 255]))

    if mn < scale_range[0] or mx > scale_range[1]:
        return (scale_range[1] - scale_range[0]) * (arr - mn) / (mx - mn) + scale_range[0]

    return arr


def channel_hist_match(source, template, hist_match_threshold=255, mask=None):
    # Code borrowed from:
    # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    masked_source = source
    masked_template = template

    if mask is not None:
        masked_source = source * mask
        masked_template = template * mask

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    masked_source = masked_source.ravel()
    masked_template = masked_template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    ms_values, mbin_idx, ms_counts = np.unique(source, return_inverse=True,
                                               return_counts=True)
    mt_values, mt_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles = hist_match_threshold * s_quantiles / s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles = 255 * t_quantiles / t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def color_hist_match(src_im, tar_im, hist_match_threshold=255):
    h, w, c = src_im.shape
    matched_R = channel_hist_match(src_im[:, :, 0], tar_im[:, :, 0], hist_match_threshold, None)
    matched_G = channel_hist_match(src_im[:, :, 1], tar_im[:, :, 1], hist_match_threshold, None)
    matched_B = channel_hist_match(src_im[:, :, 2], tar_im[:, :, 2], hist_match_threshold, None)

    to_stack = (matched_R, matched_G, matched_B)
    for i in range(3, c):
        to_stack += (src_im[:, :, i],)

    matched = np.stack(to_stack, axis=-1).astype(src_im.dtype)
    return matched
