import numpy as np


class ImageMerge():
    """
    Add the ellipsoid and its power spectrum also add some noise
    """
    def add_image(self, img_a, img_b):
        """
        combine two color image in form of nd_array side by side
        """
        ha, wa = img_a.shape[:2]                            # hight and width of img_a
        hb, wb = img_b.shape[:2]                            # hight and width of img_b
        max_height = np.max([ha, hb])                       # maximum height
        total_width = wa+wb                                 # total width
        new_img = np.zeros(shape=(max_height, total_width)) # new image of shape, maximum height and total width
        new_img[:ha, :wa] = img_a                           # insert first image
        new_img[:hb, wa:wa+wb] = img_b                      # insert second image
        return new_img

