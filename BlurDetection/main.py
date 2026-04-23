from imutils import paths
import argparse
import cv2

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True)
ap.add_argument("-t", "--threshold", type=float, default=400.0)
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["images"]))

print("Image folder:", args["images"])
print("Number of images found:", len(imagePaths))

if len(imagePaths) == 0:
    print("No images found. Check your folder path.")
    exit()

for imagePath in imagePaths:
    print("Processing:", imagePath)

    image = cv2.imread(imagePath)

    if image is None:
        print("Could not load:", imagePath)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)

    text = "Blurry" if fm < args["threshold"] else "Not Blurry"

    print(f"Result: {text} ({fm:.2f})")
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()