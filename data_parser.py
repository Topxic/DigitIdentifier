import numpy as np

# Data provided by http://yann.lecun.com/exdb/mnist/
def parse_training_data():
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

    return images, labels


def parse_test_data():
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

    return images, labels
