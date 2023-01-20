compoundA = [[['NH2', 1211.0], ['NH', 0.0], ['NH2', 1211.0]], [['DETA', 1]], [1211]]

NCA = [[[group[0], group[1] / compoundA[2][0]] if group[1] != 0 else group for group in compoundA[0]], compoundA[1], [compoundA[2][0] - 1] if compoundA[2][0] != 0 else [0]]
NCA = [[[group[0], group[1] * NCA[2][0]] if group[1] != 0 else group for group in NCA[0]], NCA[1], [NCA[2][0]] if NCA[2][0] != 0 else [0]]

print(NCA)