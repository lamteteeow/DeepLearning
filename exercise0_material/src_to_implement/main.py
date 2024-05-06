import pattern as pt


def main():
    ch = pt.Pattern(512, 64)
    ch.draw()
    ch.show()

    s = pt.Pattern(512)
    s.draw()
    s.show()

    c = pt.Pattern(512, 100, (250, 250))
    c.draw()
    c.show()


main()
