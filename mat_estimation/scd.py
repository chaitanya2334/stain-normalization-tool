import cv2
import numpy as np

from training.scd_trainer import SCDTrainer


def est_using_scd(img, trainer):
    assert isinstance(trainer, SCDTrainer)
    prob_maps, _ = trainer.classify_stain_regions(img)

    double_img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    col_img = double_img.reshape(-1, 3)

    stain_lbls = trainer.labels

    prob_maps = prob_maps.reshape(-1, len(stain_lbls))

    # background probability threshold
    tbg = 0.75

    # stain probability threshold
    tfg = 0.75

    # label used by classifier for background pixels should always be zero
    bg_lbl = 0
    bg_idx = -1

    labels = -np.ones((col_img.shape[0], 1))

    for i in range(stain_lbls):
        if stain_lbls(i) == bg_lbl:
            bg_idx = i
        else:
            # Set the label to the current stain's label for all pixels with a
            # classification probability above the stain threshold
            labels[prob_maps[:, i] > tfg] = stain_lbls[i]

    stain_lbls = stain_lbls[stain_lbls != bg_lbl]

    if bg_idx != -1:
        labels[prob_maps[:, bg_idx] > tbg] = bg_lbl

    # Generate Stain separation matrix
    m = np.zeros(3)

    m[0, :] = -np.log(np.mean(col_img[labels == stain_lbls[1], 1:3]) + (1 / 256))
    m[1, :] = -np.log(np.mean(col_img[labels == stain_lbls[0], 1:3]) + (1 / 256))

    # Third stain vector is computed as a cross product of the first two
    m[2, :] = np.cross(m[0, :], m[1, :])

    m = m / np.tile(np.sqrt(np.sum(m ^ 2, 2)), [1, 3])

    labels = labels.reshape(img.shape[0], img.shape[1])
