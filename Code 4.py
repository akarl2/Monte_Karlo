import numpy as np

# Example arrays
selected_groups = np.array([['NH', 'COOH'], ['COOH', 'POH'],['C', 'B']])
rxn_groups = np.array([['NH', 'COOH'], ['COOH', 'POH'], ['A', 'B']])

print(selected_groups)

rxn_mask = np.all(selected_groups == rxn_groups, axis=1)

print(rxn_mask)
