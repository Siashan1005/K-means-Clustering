from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from matplotlib.image import imread

def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # TODO: Implement init_centroids
    # *** START YOUR CODE ***
    # raise NotImplementedError('init_centroids function not implemented')
    # *** END YOUR CODE ***
    candidates = image.reshape(-1, image.shape[-1])
    center_indeices = np.random.choice(candidates.shape[0], size = num_clusters, replace=False)
    centroids_init = candidates[center_indeices]
    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # TODO: Implement update_centroids
    # *** START YOUR CODE ***
    # raise NotImplementedError('update_centroids function not implemented')
        # Usually expected to converge long before `max_iter` iterations
                # Initialize `dist` vector to keep track of distance to every centroid
                # Loop over all centroids and store distances in `dist`
                # Find closest centroid and update `new_centroids`
        # Update `new_centroids`
    # *** END YOUR CODE ***
    pixels = image.reshape(-1, image.shape[-1])  # Flatten the image into (N, 3)

for iteration in range(max_iter):
    # Calculate distances from each pixel to each centroid
    distances = np.linalg.norm(pixels[:, None] - centroids, axis=2)
    
    # Assign each pixel to the nearest centroid
    closest_centroids = np.argmin(distances, axis=1)
    
    # Update centroids based on the mean of assigned pixels
    new_centroids = np.array([
        pixels[closest_centroids == k].mean(axis=0) if np.any(closest_centroids == k) else centroids[k]
        for k in range(centroids.shape[0])
    ])
    
    # Check for convergence
    if np.allclose(centroids, new_centroids, atol=1e-5):
        break
    
    centroids = new_centroids

return centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # TODO: Implement update_image
    # *** START YOUR CODE ***
    # raise NotImplementedError('update_image function not implemented')
            # Initialize `dist` vector to keep track of distance to every centroid
            # Loop over all centroids and store distances in `dist`
            # Find closest centroid and update pixel value in `image`
    # *** END YOUR CODE ***

flattened_image = image.reshape(-1, image.shape[-1])
distance_matrix = np.linalg.norm(flattened_image[:, np.newaxis] - centroids, axis=2)
nearest_centroid_indices = np.argmin(distance_matrix, axis=1)
quantized_pixels = centroids[nearest_centroid_indices]
quantized_image = quantized_pixels.reshape(image.shape)

return quantized_image



def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # TODO: Load small image
    # *** START YOUR CODE ***
    image_small = imread(image_path_small)
    # *** END YOUR CODE ***

    # TODO: Initialize centroids
    # *** START YOUR CODE ***
    centroids = init_centroids(num_clusters, image_small)
    # *** END YOUR CODE ***

    # TODO: Update centroids
    # *** START YOUR CODE ***
    centroids = update_centroids(centroids, image_small, max_iter=max_iter, print_every=print_every)
    # *** END YOUR CODE ***

    # TODO: Load large image
    # *** START YOUR CODE ***
    image_large = imread(image_path_large)
    # *** END YOUR CODE ***

    # TODO: Update large image with centroids calculated on small image
    # *** START YOUR CODE ***
    compressed_image = update_image(image_large, centroids)
    # *** END YOUR CODE ***

    # TODO: Visualize and save compressed image
    # *** START YOUR CODE ***
    # Display the original large image

plt.figure(figure_idx)
plt.title('Original Image')
plt.imshow(original_image)
plt.axis('off')
figure_idx += 1

plt.figure(figure_idx)
plt.title('Quantized Image')
plt.imshow(quantized_image.astype(np.uint8))
plt.axis('off')
figure_idx += 1

output_file = 'quantized_image_large.png'
plt.imsave(output_file, quantized_image.astype(np.uint8))
print(f"Quantized image saved as: {output_file}")

    # *** END YOUR CODE ***

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
