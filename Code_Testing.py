import random

list1 = [[['NH₂', 1.0], ['NH', 0.0], ['NH₂', 1.0]], [103.169], [['DETA', 1]], [1211]]
list2 = [[['COOH', 1.0]], [282.47], [['C181', 1]], [442]]

NH2 = []
COOH = []

for i in list1[0]:
    if i[0] == 'NH₂':
        NH2.append(i + [list1[3]])

for i in list2[0]:
    if i[0] == 'COOH':
        COOH.append(i + [list2[3]])

print(NH2)
print(COOH)

NH2_weighted = []
COOH_weighted = []

for i in NH2:
    for _ in range(i[2]):
        NH2_weighted.append(i[:2])

for i in COOH:
    for _ in range(i[2]):
        COOH_weighted.append(i[:2])

random_NH2 = random.choices(NH2_weighted, weights=[i[1] for i in NH2_weighted])
random_COOH = random.choices(COOH_weighted, weights=[i[1] for i in COOH_weighted])

combined_list = [random_NH2[0], random_COOH[0]]

print(combined_list)
