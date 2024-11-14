import skimage.io as io
import matplotlib.pyplot as plt

img = io.imread('data/cell_3d.tiff')

fig, ax = plt.subplots(figsize=[10,10])
ax.imshow(img[0])
plt.show()