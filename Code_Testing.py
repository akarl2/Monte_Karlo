CA = [['DETA', 1]]
CC = [['DETA', 1], ['COOH', 1], ['COC', 1], ['Cl', 1]]
CB = [['COOH', 1]]


#combine CB and CC by adding the values of the second list to the first list and create a new list with the results

groups = {}
for group, count in CC + CB:
    if group in groups:
        groups[group] += count
    else:
        groups[group] = count

combined_list = [[group, count] for group, count in groups.items()]

print(combined_list)
