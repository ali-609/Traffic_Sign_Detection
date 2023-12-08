import matplotlib.pyplot as plt
import matplotlib.image as mpimg

input_image = mpimg.imread('input.png')
mask_image = mpimg.imread('mask.png')
output_image = mpimg.imread('output.png')

# Create subplots to display the images side by side
plt.figure(figsize=(10, 4))

# Plot the input image
plt.subplot(1, 3, 1)
plt.imshow(input_image)
plt.title('Input Image')
plt.axis('off')

# Plot the mask image
plt.subplot(1, 3, 2)
plt.imshow(mask_image)
plt.title('Mask Image')
plt.axis('off')

# Plot the output image
plt.subplot(1, 3, 3)
plt.imshow(output_image)
plt.title('Output Image')
plt.axis('off')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()