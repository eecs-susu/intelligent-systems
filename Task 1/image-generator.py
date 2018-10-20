import os

from random import Random

import cv2
import numpy as np


def generate_point(width, height, random):
    x = np.float32(random.uniform(0, width))
    y = np.float32(random.uniform(0, height))
    return x, y


def generate_circle_coords(width, height, random, padding=1):
    upper_radius = min(width, height) / 2
    radius = np.float32(random.uniform(10.0, upper_radius))
    margin = 2 * radius + padding
    shift = padding + radius
    center_x = np.float32(shift + random.uniform(0, width - margin))
    center_y = np.float32(shift + random.uniform(0, height - margin))
    return center_x, center_y, radius


def generate_ellipse_coords(width, height, random, padding=1):
    x, y, ra = generate_circle_coords(width, height, random, padding)
    rb = np.float32(random.uniform(ra / 2, ra - 1.0))
    return x, y, ra, rb


def generate_circles(width, height, count=1, seed=None):
    random = Random()
    random.seed(seed)

    matrix = np.zeros((height, width), np.float32)
    source = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    for i in range(count):
        image = source.copy()
        x, y, r = generate_circle_coords(width, height, random)
        yield cv2.circle(image, (x, y), r, (255, 255, 255))


def generate_ellipses(width, height, count=1, seed=None):
    random = Random()
    random.seed(seed)

    matrix = np.zeros((height, width), np.float32)
    source = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    for i in range(count):
        image = source.copy()
        x, y, ra, rb = generate_ellipse_coords(width, height, random)
        angle = random.randint(0, 360)
        color = (255, 255, 255)
        yield cv2.ellipse(image, (x, y), (ra, rb), angle, 0, 360, color)


def generate_broken_lines(width, height, count=1, seed=None):
    random = Random()
    random.seed(seed)

    matrix = np.zeros((height, width), np.float32)
    source = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    for i in range(count):
        image = source.copy()
        start = generate_point(width, height, random)
        mid = generate_point(width, height, random)
        end = generate_point(width, height, random)
        image = cv2.line(image, start, mid, (255, 255, 255))
        yield cv2.line(image, mid, end, (255, 255, 255))


def generate_lines(width, height, count=1, seed=None):
    random = Random()
    random.seed(seed)

    matrix = np.zeros((height, width), np.float32)
    source = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    for i in range(count):
        image = source.copy()
        start = generate_point(width, height, random)
        end = generate_point(width, height, random)
        yield cv2.line(image, start, end, (255, 255, 255))


def save_shapes(path, generator):

    if not os.path.exists(path):
        os.makedirs(path)

    for idx, image in enumerate(generator):
        cv2.imwrite(os.path.join(path, '{}.bmp'.format(idx)), image)


def main():
    path = 'images'
    samples_per_type = 10
    width, heigth = 200, 200

    shapes = [
        ('circles', generate_circles),
        ('ellipses', generate_ellipses),
        ('lines', generate_lines),
        ('broken-lines', generate_broken_lines),
    ]

    for sub_path, generator in shapes:
        save_shapes(os.path.join(path, sub_path),
                    generator(width, heigth, samples_per_type, 100))


if __name__ == '__main__':
    main()
