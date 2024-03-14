test = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
not_equal_to_g = []
equal_to_g = False

while not equal_to_g:
    for i in test:
        if i == "g":
            equal_to_g = True
        else:
            not_equal_to_g.append(i)
            break

