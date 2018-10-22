import math
import os
import sys

import cv2
import numpy as np


class Line(object):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self._norm = (a**2 + b**2) ** 0.5
        self._const_arg = None
        self._const_val = None

    @classmethod
    def from_coords(cls, x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) < 1e-6:
            line = cls(1, 1, 1)
            line._const_arg = (x0 + x1) / 2
        elif abs(dy) < 1e-6:
            line = cls(1, 1, 1)
            line._const_val = (y0 + y1) / 2
        else:
            line = cls(1.0/(dx), 1.0/(-dy), y0/(dy) - x0/(dx))
        return line

    def dist_to(self, x, y):
        if self._const_arg is not None:
            return abs(x - self._const_arg)
        if self._const_val is not None:
            return abs(y - self._const_val)
        return abs(self.a * x + self.b * y + self.c) / self._norm


def is_circle(image):
    if have_solid_field(image):
        return False

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


def is_equilateral_triangle(image):
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


def is_isosceles_triangle(image):
    if not is_triangle(image):
        return False

    corner_points = parse_triangle_corners(image)
    segments = []
    for i, corner in enumerate(corner_points):
        neig = corner_points[(i + 1) % 3]
        dist = np.linalg.norm(corner - neig)
        segments.append(dist)
    combinations = []
    for i, segment in enumerate(segments):
        combinations.append((segment, segments[(i + 1) % 3]))
    possibilities = [np.std(combination) for combination in combinations]
    return any(np.array(possibilities) < 0.6)


def get_dist(a, b):
    return np.linalg.norm(a - b)


def is_rectangle(image, only_square=False):
    diag_coord = get_diameter_points(image)
    grid = cv2.circle(image.copy(), tuple(diag_coord[0][::-1]), 3, 0, -1)
    grid = cv2.circle(grid, tuple(diag_coord[1][::-1]), 3, 0, -1)
    diags_coords = [diag_coord, get_diameter_points(grid)]
    std = np.std([get_dist(*diags_coords[0]), get_dist(*diags_coords[1])])
    if std > 0.65:
        return False

    edges = [
        [diags_coords[0][0], diags_coords[1][0]],
        [diags_coords[0][0], diags_coords[1][1]],
        [diags_coords[0][1], diags_coords[1][0]],
        [diags_coords[0][1], diags_coords[1][1]],
    ]

    if only_square:
        std = np.std([get_dist(*edge) for edge in edges])
    else:
        std = np.std([get_dist(*edges[0]), get_dist(*edges[3])])
        std = max(std, np.std([get_dist(*edges[1]), get_dist(*edges[2])]))

    if std > 1.6:
        return False

    pts_number = image.nonzero()[0].shape[0]
    found_pts_number = 0
    grid = image.copy()
    for edge in edges:
        pts = [tuple(pt[::-1]) for pt in edge]
        grid = cv2.line(grid, *pts, 0, 3)
        rest = pts_number - found_pts_number
        found_pts_number += rest - grid.nonzero()[0].shape[0]
    diff = abs(pts_number - found_pts_number)

    return diff < 10


def is_square(image):
    return is_rectangle(image, True)


def is_ellipse(image):
    if have_solid_field(image):
        return False

    c, da, db = parse_triangle_corners(image)
    line = Line.from_coords(*da, *db)
    b = int(round(line.dist_to(*c)))
    center = (da + db) / 2
    a = int(round(get_dist(da, center)))
    center = tuple(center[::-1].astype(np.int))
    axes = (a, b)
    line = Line.from_coords(*(da[::-1]), *(db[::-1]))
    tg = -line.a/line.b
    ang = int(round(180 + math.atan(tg)*180/math.pi)) % 180

    grid = image.copy()
    grid = cv2.ellipse(grid, center, axes, ang, 0, 360, 0, 2)

    pts_number = image.nonzero()[0].shape[0]
    left_number = grid.nonzero()[0].shape[0]

    ratio = left_number / pts_number
    return ratio < 0.4


def main():
    path = 'images'

    shapes = [
        'circles',
        'ellipses',
        'lines',
        'broken-lines',
        'rectangles',
        'rotated-rectangles',
        'squares',
        'right-triangles',
        'isosceles-triangles',
        'equilateral-triangles',
        'rotated-squares',
    ]

    detectors = [
        ('Circle', is_circle),
        ('Line', is_line),
        ('Broken line', is_broken_line),
        ('Triangle', is_triangle),
        ('Right triangle', is_right_triangle),
        ('Equilateral triangle', is_equilateral_triangle),
        ('Isosceles triangle', is_isosceles_triangle),
        ('Rectangle', is_rectangle),
        ('Square', is_square),
        ('Ellipse', is_ellipse),
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
