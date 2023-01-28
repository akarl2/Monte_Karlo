

test = [[[['NH2', 1.0], ['NH', 0.0], ['NH2', 1.0]], 'DETA(1)', 103.169, 800], [[['CONH', 0.0], ['NH', 0.0], ['NH2', 1.0]], 'C14(1)_DETA(1)', 313.53, 372], [[['CONH', 0.0], ['CONH', 0.0], ['NH', 0.0]], 'C14(2)_DETA(1)', 523.89, 39], [[['COOH', 1.0]], 'C14(1)', 228.376, 97]]


test_tuple = tuple([(tuple([tuple(x) for x in i[0]]), i[1], i[2], i[3]) for i in test])
# convert test to a tuples of tuples

print(test_tuple)
