from pattern import Checker, Circle, Spectrum


def main():
    checker = Checker(1024, 128)
    checker.draw()
    checker.show()

    circle = Circle(1024, 40, (256, 256))
    circle.draw()
    circle.show()

    spectrum = Spectrum(1024)
    spectrum.draw()
    spectrum.show()


if __name__ == "__main__":
    main()
