from skimage import io, filters
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Get the path to the image
    image_path = input("Enter the path to the image: ")

    try:
        # Load the image
        image = io.imread(image_path)

        # Apply Gaussian blur
        blurred_image = filters.gaussian(image, sigma=2)  # You can adjust the sigma value

        # Display the original and blurred images side by side
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(blurred_image)
        plt.title("Blurred Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Save the blurred image
        save_path = "blurred_image.jpg"
        io.imsave(save_path, np.uint8(blurred_image))

        print("Blurred image saved as", save_path)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
