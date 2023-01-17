import random

l1 = [[['NH₂', 1.0], ['NH', 0.0], ['NH₂', 1.0]], [103.169], [['DETA', 1]], [6058]]
l2 = [[['NH₂', 1.0], ['OH', 0.0], ['NH₂', 1.0]], [105.14], [['DEA', 1]], [5944]]
l3 = [[['COOH', 1.0]], [282.47], [['C181', 1]], [2212]]
l4 = [[['COOH', 1.0]], [284.48], [['ISA', 1]], [2197]]


last_value = l1[-1][0] # extract the last value, 6058

result = []
cv = 0
for group in l1[0]:

    weight = group[1]
    cv = group[1] + weight
    result.append(weight * last_value + cv)

print(result)

#determine the with the largest value, multiply by the last value in the list by the weights.  group can be selected at random by using that value

