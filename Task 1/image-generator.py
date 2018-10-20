import os

from random import Random

import cv2
import numpy as np

COLOR_WHITE = (255, 255, 255)


def generate_point(width, height, random, padding=0):
    x = np.float32(random.uniform(5, width - padding))
    y = np.float32(random.uniform(5, height - padding))
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
        yield cv2.circle(image, (x, y), r, COLOR_WHITE,
                         lineType=cv2.LINE_AA)


def generate_ellipses(width, height, count=1, seed=None):
    random = Random()
    random.seed(seed)

    matrix = np.zeros((height, width), np.float32)
    source = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    for i in range(count):
        image = source.copy()
        x, y, ra, rb = generate_ellipse_coords(width, height, random)
        angle = random.randint(0, 360)
        yield cv2.ellipse(image, (x, y), (ra, rb), angle, 0, 360, COLOR_WHITE,
                          lineType=cv2.LINE_AA)


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
        image = cv2.line(image, start, mid, COLOR_WHITE)
        yield cv2.line(image, mid, end, COLOR_WHITE,
                       lineType=cv2.LINE_AA)


def generate_lines(width, height, count=1, seed=None):
    random = Random()
    random.seed(seed)

    matrix = np.zeros((height, width), np.float32)
    source = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    for i in range(count):
        image = source.copy()
        start = generate_point(width, height, random)
        end = generate_point(width, height, random)
        yield cv2.line(image, start, end, COLOR_WHITE,
                       lineType=cv2.LINE_AA)


def generate_rectangles(width, height, count=1, seed=None):
    random = Random()
    random.seed(seed)

    matrix = np.zeros((height, width), np.float32)
    source = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    for i in range(count):
        image = source.copy()
        start = generate_point(width, height, random)
        end = generate_point(width, height, random)
        yield cv2.rectangle(image, start, end, COLOR_WHITE,
                            lineType=cv2.LINE_AA)


def generate_squares(width, height, count=1, seed=None):
    random = Random()
    random.seed(seed)

    matrix = np.zeros((height, width), np.float32)
    source = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    for i in range(count):
        image = source.copy()
        dist = np.float32(random.randint(5, min(width, height)))
        start = generate_point(width, height, random, dist + 5)
        x, y = start
        yield cv2.rectangle(image, start, (x + dist, y + dist), COLOR_WHITE,
                            lineType=cv2.LINE_AA)


def generate_right_triangles(width, height, count=1, seed=None):
    random = Random()
    random.seed(seed)

    matrix = np.zeros((height, width), np.float32)
    source = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    for i in range(count):
        image = source.copy()
        x2, y2 = -1, -1
        x0, y0, x1, y1 = 0., 0., 0., 0.
        while not (0 <= x2 < width and 0 <= y2 < height):
            x0, y0 = generate_point(width, height, random)
            x1, y1 = generate_point(width, height, random)
            dx, dy = x1 - x0, y1 - y0
            x2 = np.float32(x0 + dy * random.random())
            y2 = np.float32(y0 - dx * random.random())
        image = cv2.line(image, (x0, y0), (x1, y1), COLOR_WHITE)
        image = cv2.line(image, (x2, y2), (x1, y1), COLOR_WHITE)
        yield cv2.line(image, (x0, y0), (x2, y2), COLOR_WHITE,
                       lineType=cv2.LINE_AA)


def generate_isosceles_triangles(width, height, count=1, seed=None):
    random = Random()
    random.seed(seed)

    matrix = np.zeros((height, width), np.float32)
    source = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    for i in range(count):
        image = source.copy()
        x2, y2 = -1, -1
        x0, y0, x1, y1 = 0., 0., 0., 0.
        while not (0 <= x2 < width and 0 <= y2 < height):
            x0, y0 = generate_point(width, height, random)
            x1, y1 = generate_point(width, height, random)
            dx, dy = x1 - x0, y1 - y0
            xM, yM = (x0 + x1) / 2, (y0 + y1) / 2
            mult = 1.1
            x2 = np.float32(xM + dy * mult)
            y2 = np.float32(yM - dx * mult)
        image = cv2.line(image, (x0, y0), (x1, y1), COLOR_WHITE)
        image = cv2.line(image, (x2, y2), (x1, y1), COLOR_WHITE)
        yield cv2.line(image, (x0, y0), (x2, y2), COLOR_WHITE,
                       lineType=cv2.LINE_AA)


def generate_equilateral_triangles(width, height, count=1, seed=None):
    random = Random()
    random.seed(seed)

    matrix = np.zeros((height, width), np.float32)
    source = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    for i in range(count):
        image = source.copy()
        x2, y2 = -1, -1
        x0, y0, x1, y1 = 0., 0., 0., 0.
        while not (0 <= x2 < width and 0 <= y2 < height):
            x0, y0 = generate_point(width, height, random)
            x1, y1 = generate_point(width, height, random)
            dx, dy = x1 - x0, y1 - y0
            xM, yM = (x0 + x1) / 2, (y0 + y1) / 2
            norm = (dx * dx + dy * dy)**0.5
            dx /= norm
            dy /= norm
            mult = norm * (3.0 ** 0.5) * 0.5
            x2 = np.float32(xM + dy * mult)
            y2 = np.float32(yM - dx * mult)
        image = cv2.line(image, (x0, y0), (x1, y1), COLOR_WHITE)
        image = cv2.line(image, (x2, y2), (x1, y1), COLOR_WHITE)
        yield cv2.line(image, (x0, y0), (x2, y2), COLOR_WHITE,
                       lineType=cv2.LINE_AA)


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
        ('rectangles', generate_rectangles),
        ('squares', generate_squares),
        ('right-triangles', generate_right_triangles),
        ('isosceles-triangles', generate_isosceles_triangles),
        ('equilateral-triangles', generate_equilateral_triangles),
    ]

    for sub_path, generator in shapes:
        save_shapes(os.path.join(path, sub_path),
                    generator(width, heigth, samples_per_type, 200))


if __name__ == '__main__':
    main()
