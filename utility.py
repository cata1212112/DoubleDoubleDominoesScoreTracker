from imports import *
from constants import *

def showImage(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

def maxNegativesMinPositives(arr):
    negatives = arr[arr < 0]
    positives = arr[arr > 0]

    return np.max(negatives), np.min(positives)