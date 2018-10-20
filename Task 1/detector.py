import os

import cv2
import numpy as np


def is_circle(image):
    pixels = np.vstack(image.nonzero()).transpose()

    def get_max_dist(pt):
        return max([np.linalg.norm(pt - p) for p in pixels])
    dists = [get_max_dist(p) for p in pixels]
    return np.std(dists) < 1


def is_line(image):
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


def is_broken_line(image):
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
    return not is_line(image)


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
