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

    res = []
    for i, l in zip(images, labels):
        res.append((i, l))
    return res


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

    res = []
    for i, l in zip(images, labels):
        res.append((i, l))
    return res


def random_transform(matrix, scale_range=(0.4, 1.3), rotation_range=(-45, 45), translation_range=(-5, 5), noise_range=(0, 0.2)):
    # Get dimensions of the input matrix
    rows, cols = matrix.shape
    center = np.array([rows // 2 - 1, cols // 2 - 1, 0])

    scale = np.random.uniform(scale_range[0], scale_range[1])
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    translation = np.random.uniform(translation_range[0], translation_range[1], size=2)
    noise = np.random.uniform(noise_range[0], noise_range[1])

    rad = np.radians(rotation)
    c, s = np.cos(rad), np.sin(rad)
    T = np.array([[c*scale, -s*scale, translation[0]],
                  [s*scale, c*scale, translation[1]],
                  [0, 0, 1]])

    output_matrix = np.zeros_like(matrix)
    for r in range(rows):
        for c in range(cols):
            v = np.array([r, c, 1]) - center
            v_out = T @ v + center
            x, y = round(v_out[0]), round(v_out[1])
            if 0 <= x < rows and 0 <= y < cols:
                output_matrix[r, c] = matrix[x, y]

    return output_matrix


def training_data_generator(original_data):
    """
    Parses test data on first call. Every following call
    performs a random transformation on the original data.
    """
    i = 0
    while True:
        for img, label in original_data:
            if i == 0:
                yield (img, label)
            else:
                yield (random_transform(img), label)
        i += 1
