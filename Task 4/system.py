from engine import FourStrokeEngine


def main():
    engine = FourStrokeEngine(4)
    for stroke in range(4):
        print('Такт: {}'.format(stroke + 1))
        engine.process_stroke()
        print()


if __name__ == '__main__':
    main()
