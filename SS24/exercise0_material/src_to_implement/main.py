import generator as gen


def main():
    # ch = pt.Pattern(512, 64)
    # ch.draw()
    # ch.show()

    # s = pt.Pattern(512)
    # s.draw()
    # s.show()

    # c = pt.Pattern(512, 100, (50, 50))
    # c.draw()
    # c.show()

    i = gen.ImageGenerator(
        "./exercise_data/",
        "./Labels.json",
        32,
        (32, 32, 3),
        False,
        False,
        False,
    )
    i.show()


main()
