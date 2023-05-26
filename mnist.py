from matplotlib import pyplot as plt
import numpy as np

class MNISTSample:
    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.imageVec = image.flatten() / 255
        self.labelVec = np.zeros(10)
        self.labelVec[label] = 1

    def transform(self, scale, rotation, translation, noise):
        rows, cols = self.image.shape
        center = np.array([rows // 2 - 1, cols // 2 - 1, 0])

        rad = np.radians(rotation)
        c, s = np.cos(rad), np.sin(rad)
        T = np.array([[c*scale, -s*scale, translation[0]],
                      [s*scale, c*scale, translation[1]],
                      [0, 0, 1]])

        outputImage = np.zeros_like(self.image)
        for r in range(rows):
            for c in range(cols):
                v = np.array([r, c, 1]) - center
                v_out = T @ v + center
                x, y = round(v_out[0]), round(v_out[1])
                if 0 <= x < rows and 0 <= y < cols:
                    outputImage[r, c] = self.image[x, y]

        noiseImage = noise * np.random.rand(rows, cols)

        return np.minimum(255, outputImage + noiseImage)


# Data provided by http://yann.lecun.com/exdb/mnist/
def parseTrainingData():
    with open("./train-images.idx3-ubyte", "rb") as f:
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        image_count = int.from_bytes(f.read(4), byteorder="big")
        rows = int.from_bytes(f.read(4), byteorder="big")
        columns = int.from_bytes(f.read(4), byteorder="big")
        data = np.fromfile(f, dtype=np.uint8)
        images = data.reshape((image_count, rows, columns))

    with open("./train-labels.idx1-ubyte", "rb") as f:
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        label_count = int.from_bytes(f.read(4), byteorder="big")
        labels = np.fromfile(f, dtype=np.uint8)

    trainingSamples = []
    for image, label in zip(images, labels):
        trainingSamples += [MNISTSample(image, label)]
    return trainingSamples


def parseTestData():
    with open("./t10k-images.idx3-ubyte", "rb") as f:
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        image_count = int.from_bytes(f.read(4), byteorder="big")
        rows = int.from_bytes(f.read(4), byteorder="big")
        columns = int.from_bytes(f.read(4), byteorder="big")
        data = np.fromfile(f, dtype=np.uint8)
        images = data.reshape((image_count, rows, columns))

    with open("./t10k-labels.idx1-ubyte", "rb") as f:
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        label_count = int.from_bytes(f.read(4), byteorder="big")
        labels = np.fromfile(f, dtype=np.uint8)

    testSamples = []
    for image, label in zip(images, labels):
        testSamples += [MNISTSample(image, label)]
    return testSamples
