# import necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2

class CCluster:
    def __init__(self, clusters):
        self.clusters = clusters

    def fit(self, image):
        # reshape the image to be a single line instead of MxN matrix
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        # clustering the colors
        clt = KMeans(n_clusters=self.clusters)
        clt.fit(image)

        # return the clusters
        return clt

    def plot(self, image):
        # convert the image to RGB and show it
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.axis("off")
        plt.imshow(image)

    def bar(self, image):
        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color
        clt = self.fit(image)
        hist = self.centroid_histogram(clt)
        bar = self.plot_colors(hist, clt.cluster_centers_)

        #show the color bar
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()

    def centroid_histogram(self, clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        # normalize the histogram
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist

    def plot_colors(self, hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype = "uint8")
        startX = 0

        # loop over the percentage of each cluster and the color of each cluster
        for (perent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar
