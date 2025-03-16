# K-means-Clustering
In this project, I implemented K-means clustering from scratch to perform image compression on two images, peppers-small.tiff and peppers-large.tiff. The goal was to reduce the number of colors in the image to k = 16, thereby significantly compressing the image while maintaining visual quality.

First, I implemented the key functions for K-means clustering, including init_centroids, update_centroids, and update_image. Using Euclidean distance, I initialized cluster centers by randomly selecting pixels from peppers-small.tiff and iteratively updated the centroids for at least 30 iterations to achieve convergence.

Next, I applied the trained centroids from the small image to compress the larger image, peppers-large.tiff, by replacing each pixel’s (red, green, blue) values with the nearest cluster centroid. I then visualized the original and compressed images using Matplotlib to compare the effectiveness of the compression.

Finally, I calculated the compression factor by determining how much the image size was reduced when representing it with only 16 colors instead of 256. This project reinforced my understanding of unsupervised learning, clustering algorithms, and their applications in image processing and data compression.
