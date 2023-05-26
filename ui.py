import pygame
import numpy as np
from activation import *
from cost import *
from layer import *
from mnist import *
from training import *

# Create network training data and train
trainingData = parseTrainingData()
scaleRange=(0.2, 1.1)
rotationRange=(-45, 45)
translationRange=(-5, 5)
noiseRange=(0, 0.8)
transformedTrainingData = []
counter = 1
for sample in trainingData[:1000]:
    for _ in range(1):
        scale = np.random.uniform(scaleRange[0], scaleRange[1])
        rotation = np.random.uniform(rotationRange[0], rotationRange[1])
        translation = np.random.uniform(translationRange[0], translationRange[1], size=2)
        noise = np.random.uniform(noiseRange[0], noiseRange[1])
        transformedImage = sample.transform(scale, rotation, translation, noise)
        transformedTrainingData += [MNISTSample(transformedImage, sample.label)]
    print(f'\rGenerating data: {counter / len(trainingData[:1000]) * 100:.3f}%', end='')
    counter += 1
trainingData += transformedTrainingData
testData = parseTestData()

network = [
    Layer(784, 10, SigmoidActivation)
]
backpropagation(network, trainingData, testData, 0.01, 1, 3, MeanSquareCost)


# Configuration
scale = 10
size = np.array([28, 28])
window_size = np.array([scale * 28 + 170, scale * 28])
canvas = np.zeros(size)

white = np.array([255, 255, 255])
red = np.array([255, 0, 0])
black = np.array([0, 0, 0])

result = np.zeros(10)

# Create GUI
pygame.init()
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Draw a Map")
font = pygame.font.SysFont('arial', 30)
clock = pygame.time.Clock()


running = True
drawing = False
while running:
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                canvas = np.zeros((28, 28))
                result = np.zeros(10)
            if event.key == pygame.K_s:
                result = network.eval(canvas.flatten())


    # get the position of the mouse
    mouse_pos = np.array(pygame.mouse.get_pos())
    canvas_pos = (mouse_pos / scale).astype(np.uint)

    # Color neighbouring cells based on the distance to the mouse
    if drawing:
        cell_neighbourhood = [canvas_pos + np.array([x, y]) for x in [-1, 0, 1] for y in [-1, 0, 1]]
        for cell in cell_neighbourhood:
            if not (0 <= cell[0] < 27 and 0 <= cell[1] < 27):
                continue
            cell_center = scale * cell + scale // 2
            distance = np.sqrt(np.sum(np.square(mouse_pos - cell_center)))

            idx = int(cell[1]), int(cell[0])
            val = canvas[idx]
            val += np.exp(-distance + (0.6 * scale))
            canvas[idx] = max(min(val, 1), 0)   
        x = canvas.flatten() 
        for layer in network:
            x = layer.forward(x)
        result = x

    # Draw canvas on screen
    screen.fill(white)
    screen.fill(black, (0, 0, scale * 28, scale * 28))
    for y in range(size[1]):
        for x in range(size[0]):
            val = canvas[y, x]
            if val > 0:
                pygame.draw.rect(screen, val * white, (x * scale, y * scale, scale, scale))

    # draw the instructions
    for i in range(result.shape[0]):
        if i == np.argmax(result):
            instructions_text = font.render(f"{i}: {result[i] * 100:.3f}%", True, red)
        else:
            instructions_text = font.render(f"{i}: {result[i] * 100:.3f}%", True, black)
        screen.blit(instructions_text, (scale * 30 + 5, 28 * i))

    pygame.display.update()
    clock.tick(60)

# quit pygame
pygame.quit()