import utils
import numpy as np


def deconvolve(img, stain_mat=None):
    utils.check_rgb(img)

    h, w, c = img.shape

    # Default Color Deconvolution Matrix proposed in Ruifork and Johnston
    if not stain_mat:
        stain_mat = np.array([[0.644211, 0.716556, 0.266844],
                              [0.092789, 0.954111, 0.283111]])

    # Add third Stain vector, if only two stain vectors are provided.
    # This stain vector is obtained as the cross product of first two
    # stain vectors
    if stain_mat.shape[0] < 3:
        np.stack((stain_mat, np.cross(stain_mat[0, :], stain_mat[1, :])), axis=2)

    # Normalise the input so that each stain vector has a Euclidean norm of 1
    stain_mat = (stain_mat / np.tile(np.sqrt(np.sum(stain_mat ^ 2, 2)), [1, 3]))

    # MAIN IMPLEMENTATION OF METHOD

    # the intensity of light entering the specimen (see section 2a of [1])
    io = 255

    # Vectorize
    j_mat = img.reshape(-1, 3)

    # calculate optical density
    od_mat = -np.log((j_mat+1)/io)
    y_mat = od_mat.reshape(-1, 3)

    # determine concentrations of the individual stains
    # stain_mat is 3 x 3,  y_mat is N x 3, c_mat is N x 3
    c_mat = y_mat / stain_mat
    # c_mat = y_mat * pinv(stain_mat);

    # Stack back deconvolved channels
    deconv_ch = c_mat.reshape(h, w, 3)

    return deconv_ch, stain_mat
