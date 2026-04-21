from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser() #This initializes an object (ap) 
                            # that will collect and interpret arguments 
                            # passed from the command line.
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
                            # -i → short version of the argument
                            # --image → full version
                            # required=True → the user must provide this argument
                            # help=... → description shown if the user runs --help

                            # This argument expects a string, typically a file 
                            # path to an image.
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
                            # Again, both short and long versions required=True → must be provided
                            # type=int → converts the input into an integer 
                            # This is likely used for something like clustering (e.g., k-means), 
                            # where you specify how many clusters to use.
args = vars(ap.parse_args()) # ap.parse_args() reads the arguments entered in the terminal
                            # vars(...) converts the result into a dictionary
                            

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show our image
plt.figure()
plt.axis("off")
plt.imshow(image)

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

# cluster the pixel intensities
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)


# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)
# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()