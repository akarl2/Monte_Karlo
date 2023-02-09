composition = [[[['NH2', 1.0], ['NH', 0.0], ['NH2', 1.0]], [['DETA', 1]], [103.169]], [[['NH2', 1.0], ['NH', 0.0], ['NH2', 1.0]], [['DETA', 1]], [103.169]], [[['NH2', 1.0], ['NH', 0.0], ['NH2', 1.0]], [['DETA', 1]], [103.169]], [[['NH2', 1.0], ['NH', 0.0], ['NH2', 1.0]], [['DETA', 1]], [103.169]], [[['NH2', 1.0], ['NH', 0.0], ['NH2', 1.0]], [['DETA', 1]], [103.169]], [[['NH2', 1.0], ['NH', 0.0], ['NH2', 1.0]], [['DETA', 1]], [103.169]], [[['COOH', 1.0]], [['C14', 1]], [228.376]], [[['COOH', 1.0]], [['C14', 1]], [228.376]]]
chemical = [[0, 0, 'NH2'], [0, 1, 'NH'], [0, 2, 'NH2'], [1, 0, 'NH2'], [1, 1, 'NH'], [1, 2, 'NH2'], [2, 0, 'NH2'], [2, 1, 'NH'], [2, 2, 'NH2'], [3, 0, 'NH2'], [3, 1, 'NH'], [3, 2, 'NH2'], [4, 0, 'NH2'], [4, 1, 'NH'], [4, 2, 'NH2'], [5, 0, 'NH2'], [5, 1, 'NH'], [5, 2, 'NH2'], [6, 0, 'COOH'], [7, 0, 'COOH']]
weights = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]

low_group = 4
weights = []
chemical = []
for i, chemicals in enumerate(composition):
    index = 0
    for group in chemicals[0]:
        chemical.append([i, index, group[0]])
        weights.append(group[1])
        index += 1

#starting when chemica

print(chemical)
print(weights)