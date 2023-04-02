def distance(a, b):
    ab0 = a[0] - b[0]
    ab1 = a[1] - b[1]
    ab2 = a[2] - b[2]
    return ab0 * ab0 + ab1 * ab1 + ab2 * ab2

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a