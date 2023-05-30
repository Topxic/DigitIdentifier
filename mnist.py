import os
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

SCALE_RANGE = (0.7, 1.1)
ROTATION_RANGE = (-20, 20)
TRANSLATION_RANGE = (-5, 5)
NOISE_RANGE = (0.05, 0.1)

FILE_NAME = 'mnist-training-data.pkl'

class MNISTSample:
    def __init__(self, image: np.ndarray, label: int):
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
        scale = 1 / scale
        T = np.array([[c*scale, -s*scale, translation[1]],
                      [s*scale, c*scale, translation[0]],
                      [0, 0, 1]])

        outputImage = np.zeros_like(self.image)
        for r in range(rows):
            for c in range(cols):
                v = np.array([r, c, 1]) - center
                v_out = T @ v + center
                x, y = round(v_out[0]), round(v_out[1])
                if 0 <= x < rows and 0 <= y < cols:
                    outputImage[r, c] = self.image[x, y]

        noiseImage = noise * np.random.randn(rows, cols) * 255

        return np.clip(outputImage + noiseImage, 0, 255)
    
    def transfromRandomly(self):
        scale = np.random.uniform(SCALE_RANGE[0], SCALE_RANGE[1])
        rotation = np.random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1])
        translationX = np.random.uniform(TRANSLATION_RANGE[0], TRANSLATION_RANGE[1])
        translationY = np.random.uniform(TRANSLATION_RANGE[0], TRANSLATION_RANGE[1])
        noise = np.random.uniform(NOISE_RANGE[0], NOISE_RANGE[1])
        image = self.transform(scale, rotation, np.asarray([translationX, translationY]), noise)
        return MNISTSample(image, self.label)


# Data provided by http://yann.lecun.com/exdb/mnist/
def parseTrainingData() -> list[MNISTSample]:
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


def parseTestData() -> list[MNISTSample]:
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


def exploreTransformationParams():
    """ 
    Used for adjusting random transformation parameters 
    such that every digit is still recognizable
    """

    data = parseTrainingData()
    imgIdx = 0
    initScale = 1
    initRotation = 0
    initTranslationX = 0
    initTranslationY = 0
    initNoise = 0

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    imageContainer = plt.imshow(data[imgIdx].image, cmap='gray')
    fig.subplots_adjust(bottom=0.35)

    axfreq1 = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    scaleSlider = Slider(
        ax=axfreq1,
        label='Scale',
        valmin=SCALE_RANGE[0],
        valmax=SCALE_RANGE[1],
        valinit=initScale,
    )

    axfreq2 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    rotationSlider = Slider(
        ax=axfreq2,
        label='Rotation',
        valmin=ROTATION_RANGE[0],
        valmax=ROTATION_RANGE[1],
        valinit=initRotation,
    )

    axfreq3 = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    translationXSlider = Slider(
        ax=axfreq3,
        label='Translation X',
        valmin=TRANSLATION_RANGE[0],
        valmax=TRANSLATION_RANGE[1],
        valinit=initTranslationX,
    )

    axfreq4 = fig.add_axes([0.25, 0.2, 0.65, 0.03])
    translationYSlider = Slider(
        ax=axfreq4,
        label='Translation Y',
        valmin=TRANSLATION_RANGE[0],
        valmax=TRANSLATION_RANGE[1],
        valinit=initTranslationY,
    )

    axfreq5 = fig.add_axes([0.25, 0.25, 0.65, 0.03])
    noiseSlider = Slider(
        ax=axfreq5,
        label='Noise',
        valmin=NOISE_RANGE[0],
        valmax=NOISE_RANGE[1],
        valinit=initNoise,
    )


    # The function to be called anytime a slider's value changes
    def update(val):
        translation = np.asarray([translationXSlider.val, translationYSlider.val])
        transformedImage = data[imgIdx].transform(
            scaleSlider.val,
            rotationSlider.val, 
            translation,
            noiseSlider.val
        )
        imageContainer.set_data(transformedImage)
        plt.draw()

    # register the update function with each slider
    scaleSlider.on_changed(update)
    rotationSlider.on_changed(update)
    translationXSlider.on_changed(update)
    translationYSlider.on_changed(update)
    noiseSlider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')


    def reset(event):
        scaleSlider.reset()
        rotationSlider.reset()
        translationXSlider.reset()
        translationYSlider.reset()
        noiseSlider.reset()
    button.on_clicked(reset)

    plt.show()


def plotRandomTransformations():
    data = parseTrainingData()
    imgIdx = 0

    fig, _ = plt.subplots()
    imageContainer = plt.imshow(data[imgIdx].image, cmap='gray')
    fig.subplots_adjust(bottom=0.15)

    nextSampleAxis = fig.add_axes([0.2, 0.025, 0.25, 0.04])
    nextSampleButton = Button(nextSampleAxis, 'Next sample', hovercolor='0.975')

    randoransformationAxis = fig.add_axes([0.5, 0.025, 0.3, 0.04])
    randoransformationButton = Button(randoransformationAxis, 'Random transformation', hovercolor='0.975')

    def nextSample(event):
        nonlocal imgIdx
        imgIdx += 1
        imageContainer.set_data(data[imgIdx].image)
        plt.draw()


    def nextTransformation(event):
        imageContainer.set_data(data[imgIdx].transfromRandomly().image)
        plt.draw()


    nextSampleButton.on_clicked(nextSample)
    randoransformationButton.on_clicked(nextTransformation)

    plt.show()


def mnist2file(transformationsPerSample):
    if os.path.isfile(FILE_NAME): 
        return
    
    data = parseTrainingData() + parseTestData()
    with open(FILE_NAME, 'wb') as file:
        # Save number of samples
        pk.dump(transformationsPerSample * len(data), file)

        # Generate and save randomly transformed samples
        counter = 1
        for sample in data:
            for _ in range(transformationsPerSample):
                transformed = sample.transfromRandomly()
                pk.dump(transformed.image, file)
                pk.dump(transformed.label, file)
            print(f'\rGenerating data {counter / len(data) * 100:.3f}%', end='')
            counter += 1
    print()


def file2mnist(batchSize):
    with open(FILE_NAME, 'rb') as file:
        numSamples = pk.load(file)
        batch = []
        counter = 0
        for _ in range(numSamples):
            image = pk.load(file)
            label = pk.load(file)
            batch += [MNISTSample(image, label)]
            counter += 1

            if len(batch) == batchSize:
                yield batch, counter / numSamples
                batch = []
