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
