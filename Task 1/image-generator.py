from PIL import Image
import random


class Canvas(object):
    def __init__(self, width, height):
        self._image = Image.new('L', (width, height))

    def draw_point(self, x, y, value=255):
        if isinstance(value, float):
            value = max(0, int(value * 256) - 1)
        assert 0 <= value <= 255
        if 0 <= x < self._image.width and 0 <= y < self._image.height:
            self._image.putpixel((int(x), int(y)), value)

    def draw_line(self, x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0
        for x in range(x0, x1 + 1):
            y = y0 + dy * (x - x0) / dx
            self.draw_point(x, y)

    def draw_circle(self, x0, y0, radius):
        x = radius-1
        y = 0
        dx = 1
        dy = 1
        err = dx - (radius << 1)

        while x >= y:
            self.draw_point(x0 + x, y0 + y)
            self.draw_point(x0 + y, y0 + x)
            self.draw_point(x0 - y, y0 + x)
            self.draw_point(x0 - x, y0 + y)
            self.draw_point(x0 - x, y0 - y)
            self.draw_point(x0 - y, y0 - x)
            self.draw_point(x0 + y, y0 - x)
            self.draw_point(x0 + x, y0 - y)

            if err <= 0:
                y += 1
                err += dy
                dy += 2

            if err > 0:
                x -= 1
                dx += 2
                err += dx - (radius << 1)

    def save(self, file_name):
        self._image.save('{}.bmp'.format(file_name))


class ShapeGenerator(object):

    def __init__(self, field_size=None, seed=None):
        self._random = random.Random()
        self._random.seed(seed)
        self._width, self._height = field_size or (200, 200)

    def get_random_cirle(self):
        canvas = Canvas(self._width, self._height)
        max_rad = min(self._width, self._height) / 2
        rad = self._random.randint(1, max_rad)
        padding = 2 * rad
        x0 = self._random.randint(0, self._width - padding)
        y0 = self._random.randint(0, self._width - padding)
        canvas.draw_circle(rad + x0, rad + y0, rad)
        return canvas


def main():
    generator = ShapeGenerator(seed=42)
    circle = generator.get_random_cirle()
    circle.save('circle')


if __name__ == '__main__':
    main()
