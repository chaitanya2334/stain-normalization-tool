import numpy as np

from training.classifier_rf import ClassifierRF


class SCDTrainer(ClassifierRF):
    def __init__(self):
        super().__init__()
        self.pallet = None

    def gen_hist(self, img, pallet):
        # Generate a lookup table for the pallet colours
        # The value of lookup(R, G, B) will be the index of the colour in Pallet
        # that a pixel of value [R G B] belongs to
        lookup = np.zeros((256, 256, 256))

        # fill the lookup table
        for i in range(pallet.shape[0]):
            lookup[pallet[i, 0]: pallet[i, 1], pallet[i, 2]: pallet[i, 3], pallet[i, 4]: pallet[i, 5]] = i

        col_img = img.reshape(-1, 3)

        # Convert the pixel's RGB values to Pallet indices using the lookup table
        pallet_ind = lookup[col_img[:, 1], col_img[:, 2], col_img[:, 3]]

        # Compute the histogram from the Pallet indices
        hist = np.histogram(pallet_ind[np.all(pallet_ind != 0)], bins=range(pallet.shape[0]))

        return hist

    @staticmethod
    def build_ifv(img, scd):
        ifv = img.reshape(-1, 3)
        ifv = np.tile(scd[:].transpose(), [ifv.shape[0], 1])
        return ifv

    def classify_stain_regions(self, img):
        img = img.astype(np.uint8)

        # Find the histogram of pallet values for the input image, using the
        # precalculated pallet from training
        hist = self.gen_hist(img, self.pallet)

        # normalize the histogram so that its values sum to 1
        hist = hist / np.sum(hist)

        # Compute the SCD of the input images, using the pre-calculated
        # Principal Component Histogram (PCH) from training
        scd = (hist - self.pch.h) * self.pch.e

        img = img.astype(np.double)

        # Prepare the Image Feature Vector
        # This is the input for classification
        x = self.build_ifv(img, scd)

        # Classify Input Image
        # Classifier returns probability maps for each possible stain (and
        # background).
        # TODO : implement this in ClassifierRF
        prob_maps = self.classification_function(self.classifer, x)

        # Create a label image, taking the stain with the highest probability at
        # each pixel as the label for that pixel
        classified_labels = np.amax(prob_maps, axis=1)

        classified_labels = self.labels[classified_labels]

        # Reshape the output to the dimensions of the input image
        prob_maps = prob_maps.reshape(img.shape[0], img.shape[1], -1)
        classified_labels = classified_labels.reshape(img.shape[0], img.shape[1])

        return prob_maps, classified_labels
