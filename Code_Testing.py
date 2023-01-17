import random
from itertools import chain
from tqdm import tqdm

l1 = [[['NH₂', 1.0], ['NH', 0.0], ['NH₂', 1.0]], [103.169], [['DETA', 1]], [6058]]
l2 = [[['NH₂', 1.0], ['OH', 0.5], ['NH₂', 1.0]], [105.14], [['DEA', 1]], [5944]]
l3 = [[['COOH', 1.0]], [282.47], [['C181', 1]], [2212]]
l4 = [[['COOH', 1.0]], [284.48], [['ISA', 1]], [2197]]

l1[0] = [[group[0], group[1]*l1[3][0]] for group in l1[0]]
l2[0] = [[group[0], group[1]*l2[3][0]] for group in l2[0]]
l3[0] = [[group[0], group[1]*l3[3][0]] for group in l3[0]]
l4[0] = [[group[0], group[1]*l4[3][0]] for group in l4[0]]

lists = [l1, l2, l3, l4]
weights = []
chemical = []
for i, chemicals in enumerate(lists):
    index = 0
    for group in chemicals[0]:
        chemical.append([i, index, group[0]])
        weights.append(group[1])
        index += 1
Groups = random.choices(chemical, weights, k=2)
while Groups[0][0] == Groups[1][0]:
    Groups = random.choices(chemical, weights, k=2)
print(Groups)

print(lists[Groups[0][0]])
print(lists[Groups[1][0]])





