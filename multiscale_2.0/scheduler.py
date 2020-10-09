def class_1(L):
    return L[0]


def class_2(L):
    return L[1]


def class_3(L):
    return L[2]


def class_4(L):
    return L[3]


def outside_class(L):
    return L[4]


def weekend(L):
    return L[5]


def trivial(L):
    return L[6]

def get_L(L, argument):
    switcher = {
        0: class_1(L),
        1: class_2(L),
        2: class_3(L),
        3: class_4(L),
        4: outside_class(L),
        5: weekend(L),
        6: trivial(L)
    }
    # Get the function from switcher dictionary
    specific_L = switcher.get(argument, lambda: "No layer for that reference")
    return specific_L