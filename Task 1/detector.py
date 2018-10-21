import os

import cv2
import numpy as np


class Line(object):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self._norm = (a**2 + b**2) ** 0.5

    @classmethod
    def from_coords(cls, x0, y0, x1, y1):
        return cls(1.0/(x1 - x0), 1.0/(y0 - y1), y0/(y1 - y0) - x0/(x1 - x0))

    def dist_to(self, x, y):
        return abs(self.a * x + self.b * y + self.c) / self._norm


def is_circle(image):
    pixels = np.vstack(image.nonzero()).transpose()

    def get_max_dist(pt):
        return max([np.linalg.norm(pt - p) for p in pixels])
    dists = [get_max_dist(p) for p in pixels]
    return np.std(dists) < 1


def is_line(image):
    if not have_solid_field(image):
        return False
    pixels = np.vstack(image.nonzero()).transpose().astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pixels, mean=None)
    projects = cv2.PCAProject(pixels, mean, eigenvectors)
    return np.std(projects, axis=0)[1] < 1


def bfs(i, j, image, visited):
    visited[i][j] = True
    deltas = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    queue = [(i, j)]
    while queue:
        x, y = queue.pop(0)
        for dx, dy in deltas:
            v, u = x + dx, y + dy
            if (
                0 <= v < image.shape[0] and
                0 <= u < image.shape[1] and
                not visited[v][u] and
                image[v][u] == 0
            ):
                visited[v][u] = True
                queue.append((v, u))


def have_solid_field(image):
    marked = False
    visited = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 0.0:
                continue
            if not marked:
                bfs(i, j, image, visited)
                marked = True
            elif not visited[i][j]:
                return False
    return True


def is_broken_line(image):
    return have_solid_field(image) and not is_line(image)


def get_max_dist(point, points):
    dists = [np.linalg.norm(point - p) for p in points]
    argmax = np.argmax(dists)
    return argmax, dists[argmax]


def get_diameter_points(image):
    pixels = np.vstack(image.nonzero()).transpose().astype(np.float32)
    dists = [(i, get_max_dist(point, pixels))
             for i, point in enumerate(pixels)]
    dists = sorted(dists, key=lambda e: -e[1][1])
    return pixels[dists[0][0]], pixels[dists[0][1][0]]


def parse_triangle_corners(image):
    s, e = get_diameter_points(image)
    x0, y0 = s
    x1, y1 = e
    line = Line.from_coords(x0, y0, x1, y1)
    points = np.vstack(image.nonzero()).transpose().astype(np.float32)
    amx = np.argmax([line.dist_to(x, y) for x, y in points])
    return points[amx], s, e


def is_triangle(image):
    if have_solid_field(image):
        return False

    corner_points = parse_triangle_corners(image)
    triangle_points = [tuple(pt[::-1]) for pt in corner_points]

    grid = image.copy()
    for i, pt in enumerate(triangle_points):
        neig = triangle_points[(i + 1) % 3]
        grid = cv2.line(grid, pt, neig, 0, np.float32(2.5))

    points = np.vstack(image.nonzero()).transpose().astype(np.float32)
    new_points = np.vstack(grid.nonzero()).transpose().astype(np.float32)

    ratio = len(new_points) / len(points)
    return ratio < 0.05


def is_right_triangle(image):
    if not is_triangle(image):
        return False

    corner_points = parse_triangle_corners(image)
    segments = []
    for i, corner in enumerate(corner_points):
        neig = corner_points[(i + 1) % 3]
        dist = np.linalg.norm(corner - neig)
        segments.append(dist)
    segments = sorted(segments)
    expected = (segments[0] ** 2 + segments[1] ** 2) ** 0.5
    return abs(expected - segments[2]) < 10


def is_equilateral_triangles(image):
    if not is_triangle(image):
        return False

    corner_points = parse_triangle_corners(image)
    segments = []
    for i, corner in enumerate(corner_points):
        neig = corner_points[(i + 1) % 3]
        dist = np.linalg.norm(corner - neig)
        segments.append(dist)
    std = np.std(segments - np.mean(segments))
    return std < 0.5


def main():
    path = 'images'

    shapes = [
        'circles',
        'ellipses',
        'lines',
        'broken-lines',
        'rectangles',
        'squares',
        'right-triangles',
        'isosceles-triangles',
        'equilateral-triangles',
    ]

    detectors = [
        ('Circle', is_circle),
        ('Line', is_line),
        ('Broken line', is_broken_line),
        ('Triangle', is_triangle),
        ('Right triangle', is_right_triangle),
        ('Equilateral triangle', is_equilateral_triangles),
    ]

    for shape in shapes:
        folder = os.path.join(path, shape)
        for file_ in os.listdir(folder):
            if file_.endswith(".bmp"):
                file_path = os.path.join(folder, file_)
                image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
                image = image.astype(np.float32)
                labels = [label for label, det in detectors if det(image)]
                print('{0} {1}: {2}'.format(shape, file_, ', '.join(labels)))


if __name__ == "__main__":
    main()
