# import the necessary packages

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage.color import gray2rgba
import torch

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# load the data

filename = "160727 8"

image_t = io.imread("data/" + filename + ".tif")


if image_t.ndim == 3:
    n=0
    image = image_t[n] # for video
else:
    image = image_t # for image

#fig, ax = plt.subplots()
#ax.imshow(image, cmap="viridis") # for video
#plt.show()


# normalize the image
img_norm = ((image / image.max())*np.iinfo(image.dtype).max).astype(np.uint8)
img_norm = np.expand_dims(img_norm, axis=2)
img_norm = np.repeat(img_norm, 3, axis=2)


device="cpu"
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img_norm)

print(len(masks))

fig, ax = plt.subplots()
ax.imshow(img_norm, cmap="viridis")
show_anns(masks)
plt.show()

# save data as .jpg or png

#io.imsave("data/" + filename +".png", img_norm)