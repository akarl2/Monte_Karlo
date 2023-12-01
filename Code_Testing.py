test = [[[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['SOH', 0.0], ['NH2', 1.0], ['SOH', 0.0]], [['DEA', 1]], [105.14]], [[['COOH', 1.0]], [['C14', 1]], [228.376]], [[['COOH', 1.0]], [['C14', 1]], [228.376]], [[['COOH', 1.0]], [['C14', 1]], [228.376]], [[['COOH', 1.0]], [['C14', 1]], [228.376]], [[['COOH', 1.0]], [['C14', 1]], [228.376]], [[['COOH', 1.0]], [['C14', 1]], [228.376]], [[['COOH', 1.0]], [['C14', 1]], [228.376]], [[['COOH', 1.0]], [['C14', 1]], [228.376]], [[['COOH', 1.0]], [['C14', 1]], [228.376]], [[['COOH', 1.0]], [['C14', 1]], [228.376]]]

# Remove numerical values from the first index
modified_test = [
    [
        [[element, value] if isinstance(value, str) else [element] for element, value in sublist[0]],
        sublist[1],
        sublist[2]
    ]
    for sublist in test
]

import random

print(modified_test)

# Randomly select a sublist
selected_sublist_1 = random.choice(modified_test)

# Randomly select a group from the first sublist
selected_group_1 = random.choice(selected_sublist_1[0])

# Remove the selected sublist to ensure the second selection is from a different sublist
modified_test.remove(selected_sublist_1)

# Randomly select another sublist
selected_sublist_2 = random.choice(modified_test)

# Randomly select a group from the second sublist
selected_group_2 = random.choice(selected_sublist_2[0])

selected_group_1.sort()
selected_group_2.sort()

rxn_groups = [['COOH','NH2'], ['COOH','SOH']]
constants = [1, 0.1]

Random_Group = [selected_group_1[0], selected_group_2[0]]
Random_Group.sort()

random_number = random.uniform(0, 1)


# Display the random number
print("Random Number:", random_number)
print("Selected Groups:", Random_Group)

if Random_Group in rxn_groups:
    if random_number < constants[rxn_groups.index(Random_Group)]:
        print("Reaction Occurs")
    elif random_number > constants[rxn_groups.index(Random_Group)]:
        print("Reaction does not occur")

