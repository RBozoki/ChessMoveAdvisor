import numpy as np
from PIL import Image

def load_image_as_vector(image_path):
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((100, 100))
    return np.array(image).flatten()


def divide_image_into_vectors(image_path):
    image = Image.open(image_path)
    assert image.size == (800, 800), "L'image doit être de 800x800 pixels"

    square_size = 100
    vectors = []

    for row in range(8):
        for col in range(8):
            left = col * square_size
            upper = row * square_size
            right = left + square_size
            lower = upper + square_size

            square_image = image.crop((left, upper, right, lower))

            square_vector = np.array(square_image).flatten()
            vectors.append(square_vector)

    return vectors