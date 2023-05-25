import glob
import logging
import math
import os
import random
from datetime import datetime
from functools import partial
from hashlib import md5
from multiprocessing import Lock, Pool, Value
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import cv2  # type: ignore
import h5py  # type: ignore
import numpy as np
import openslide  # type: ignore
import PIL.Image as Image  # type: ignore
import torch
from numpy.typing import NDArray
from PIL import Image
from skimage.filters import threshold_otsu  # type: ignore
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms  # type: ignore
from utils import set_logger

T = torch.Tensor

# autopep8: off
# =====================================================================================================
# PATCH GENERATOR FROM:
# https://github.com/hrzhang1123/DTFD-MIL/blob/main/Patch_Generation/gen_patch_noLabel_stride_MultiProcessing_multiScales.py
# =====================================================================================================
# autopep8: on


# ======================================    User Configuration
num_thread = 40
patch_dimension_level = 3  # 0: 40x, 1: 20x
patch_level_list = [3]  # [1,2,2]
stride = 256
psize = 256
psize_list = [256]  # [256, 192, 256]

tissue_mask_threshold = 0.8
mask_dimension_level = 5

slides_folder_dir = '/data/CAMELYON16/images'
# change the surfix '.tif' to other if necessary
slide_paths = glob.glob(os.path.join(slides_folder_dir, '*.tif'))
save_folder_dir = '/data/CAMELYON16-224-patches'
# ======================================

# mask_level: 5, 1/32


def get_roi_bounds(
    tslide: openslide.ImageSlide,
    logger: logging.Logger,
    isDrawContoursOnImages: bool = False,
    mask_level: int = 5,
    cls_kernel: int = 50,
    opn_kernel: int = 30


) -> Tuple[NDArray[np.float32], List[Any]]:
    # this reads the whole image (reading at mask level will make the whole
    # image the size of level_dimensions[mask_level])
    subSlide = tslide.read_region(
        (0, 0), mask_level, tslide.level_dimensions[mask_level])
    subSlide_np = np.array(subSlide)
    # print(f"{tslide.level_dimensions=} {tslide.level_count=}")  levels are
    # 'levels of zoom' https://openslide.org/api/python/

    # hue, saturation, value
    hsv = cv2.cvtColor(subSlide_np, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # TODO: delete the try except or retinstate it properly if it is needed
    # https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu
    hthresh = threshold_otsu(h)
    sthresh = threshold_otsu(s)
    vthresh = threshold_otsu(v)

    # not sure where these stock values came from
    minhsv = np.array([hthresh, sthresh, 70], np.uint8)
    maxhsv = np.array([180, 255, vthresh], np.uint8)
    thresh = [minhsv, maxhsv]

    # extracting the countour for tissue
    mask = cv2.inRange(hsv, thresh[0], thresh[1])

    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    # close and open to remove noise
    close_kernel = np.ones((cls_kernel, cls_kernel), dtype=np.uint8)
    image_close_img = Image.fromarray(cv2.morphologyEx(
        np.array(mask), cv2.MORPH_CLOSE, close_kernel))

    open_kernel = np.ones((opn_kernel, opn_kernel), dtype=np.uint8)
    image_open_np = cv2.morphologyEx(
        np.array(image_close_img), cv2.MORPH_OPEN, open_kernel)

    # find image contours and create bounding boxes:
    # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    contours, _ = cv2.findContours(
        image_open_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBox = [cv2.boundingRect(c) for c in contours]
    boundingBox = [sst for sst in boundingBox if sst[2] > 150 and sst[3] > 150]

    logger.info(f"boundingBox length {len(boundingBox)} {boundingBox=}")

    if isDrawContoursOnImages:
        line_color = (0, 0, 0)  # blue color code
        contours_rgb_image_np = np.array(subSlide)
        cv2.drawContours(contours_rgb_image_np, contours, -1, line_color, 50)
        contours_rgb_image_np = cv2.resize(
            contours_rgb_image_np, (0, 0), fx=0.2, fy=0.2)
        countours_rgb_image_img = Image.fromarray(
            contours_rgb_image_np.astype(np.uint8))
        countours_rgb_image_img.show()

    return image_open_np, boundingBox


def Extract_Patch_From_Slide_STRIDE(
    tslide: openslide.ImageSlide,
    logger: logging.Logger,
    tissue_mask: NDArray,
    patch_save_folder: str,
    patch_level: int,
    mask_level: int,
    patch_stride: int,
    patch_size: int,
    threshold: float,
    level_list: List[int] = [1],
    patch_size_list: List[int] = [256],
    patch_surfix: str = 'jpg',
) -> None:

    assert patch_level == level_list[0]
    assert patch_size == patch_size_list[0]

    mask_sH, mask_sW = tissue_mask.shape
    logger.info(f'tissue mask shape {tissue_mask.shape}')

    # mask level = 5, patch_level = 1, patch_size = 256
    mask_patch_size = patch_size // pow(2, mask_level - patch_level)    # 16
    mask_patch_size_square = mask_patch_size ** 2                       # 256
    mask_stride = patch_stride // pow(2, mask_level - patch_level)      # 16
    # print(f"{mask_patch_size=}, {mask_patch_size_square=} {mask_stride=}"
    print(f"{mask_level=} {patch_level=} {mask_patch_size=} {patch_stride=}")

    # mag_factor = pow(2, mask_level-patch_level)
    mag_factor = pow(2, mask_level)    # !! means 2**5
    msg = f"slide level dimensions {tslide.level_dimensions}, "
    msg += "mask patch size {mask_patch_size}, mask stride {mask_stride}, "
    msg += "mag_factor {mag_factor}"
    logger.info(msg)

    tslide_name = os.path.basename(patch_save_folder)
    num_error = 0

    for iw in range(mask_sW // mask_stride):
        for ih in range(mask_sH // mask_stride):
            ww = iw * mask_stride
            hh = ih * mask_stride
            if (
                (ww + mask_patch_size) < mask_sW and
                (hh + mask_patch_size) < mask_sH
            ):
                # the mask_patch is a downsampled version of the full patch
                # (for efficiency?), so if we pass the threshold and become
                # active, then save this patch. Otherwise it is considered
                # a blank region
                tmask = tissue_mask[hh:hh +
                                    mask_patch_size, ww: ww + mask_patch_size]
                mRatio = float(np.sum(tmask > 0)) / mask_patch_size_square

                if mRatio > threshold:
                    for sstLevel, tSize in zip(level_list, patch_size_list):
                        try:
                            tsave_folder = getFolder_name(
                                patch_save_folder, sstLevel, tSize)

                            # the image dimensiones at every level will be
                            # doubled at every step, so if we have mask level
                            # 5 then we need to double w/h 5 times.
                            sww = ww * mag_factor
                            shh = hh * mag_factor

                            cW_l0 = sww + (patch_size // 2) * \
                                pow(2, patch_level)
                            cH_l0 = shh + (patch_size // 2) * \
                                pow(2, patch_level)

                            tlW_l0 = cW_l0 - (tSize // 2) * pow(2, sstLevel)
                            tlH_l0 = cH_l0 - (tSize // 2) * pow(2, sstLevel)
                            # (x, y) tuple giving the top left pixel in the
                            # level 0 reference frame
                            tpatch = tslide.read_region(
                                (tlW_l0, tlH_l0), sstLevel, (tSize, tSize))

                            tname = f"{tslide_name}_{ww * mag_factor}_"
                            tname += f"{hh * mag_factor}_{iw}_{ih}"
                            tname += f"_WW_{mask_sW // mask_stride}_HH_"
                            tname += f"{mask_sH // mask_stride}.{patch_surfix}"
                            tpatch = tpatch.convert("RGB")
                            tpatch.save(os.path.join(tsave_folder, tname))
                            logger.info('saved patch')
                            print(".", end="\r")
                        except:  # noqa
                            num_error += 1
                            logger.warning(
                                f'slide {tslide_name} error patch {num_error}')
    if num_error != 0:
        msg = f'-----In total {num_error} error patch for slide {tslide_name}'
        logger.warning(msg)


def getFolder_name(orig_dir: str, level: float, psize: float) -> str:
    tslide = os.path.basename(orig_dir)
    folderName = os.path.dirname(orig_dir)

    subfolder_name = float(psize * level) / 256
    tfolder = os.path.join(folderName, str(subfolder_name * 10), tslide)
    return tfolder


def read_tumor_mask(mask_path: str, mask_dimension_level: int) -> NDArray[Any]:
    tmask = openslide.open_slide(mask_path)
    subMask = tmask.read_region(
        (0, 0), mask_dimension_level,
        tmask.level_dimensions[mask_dimension_level]
    )
    subMask = subMask.convert('L')
    subMask_np = np.array(subMask)

    return subMask_np


def Thread_PatchFromSlides(args: List[Union[str, logging.Logger]]) -> None:
    normSlidePath, slideName, tsave_slide_dir, logger = args

    assert isinstance(normSlidePath, str)
    assert isinstance(slideName, str)
    assert isinstance(tsave_slide_dir, str)
    assert isinstance(logger, logging.Logger)

    logger.info("making dirs")
    for tlevel, tsize in zip(patch_level_list, psize_list):
        tsave_dir_level = getFolder_name(tsave_slide_dir, tlevel, tsize)
        if not os.path.exists(tsave_dir_level):
            os.makedirs(tsave_dir_level)

    logger.info("making dirs")
    tslide = openslide.open_slide(normSlidePath)

    logger.info("getting roi bounds")
    tissue_mask, boundingBoxes = get_roi_bounds(
        tslide, logger, isDrawContoursOnImages=False,
        mask_level=mask_dimension_level
    )  # mask_level: absolute level
    tissue_mask = tissue_mask // 255

    logger.info("extracting patch")
    Extract_Patch_From_Slide_STRIDE(tslide, logger, tissue_mask,
                                    tsave_slide_dir,
                                    patch_level=patch_dimension_level,
                                    mask_level=mask_dimension_level,
                                    patch_stride=stride, patch_size=psize,
                                    threshold=tissue_mask_threshold,
                                    level_list=patch_level_list,
                                    patch_size_list=psize_list,
                                    )


def main(logger: logging.Logger) -> None:
    pool = Pool(processes=num_thread)
    arg_list: List[Any] = []

    for tSlidePath in slide_paths:
        slideName = os.path.basename(tSlidePath).split('.')[0]
        tsave_slide_dir = os.path.join(save_folder_dir, slideName)
        arg_list.append([tSlidePath, slideName, tsave_slide_dir, logger])

    pool.map(Thread_PatchFromSlides, arg_list)


if __name__ == "__main__":
    logger = set_logger("WARNING")
    main(logger)
