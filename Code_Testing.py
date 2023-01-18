

starting_materials = [[[['NH₂', 1.0], ['OH', 0.5], ['NH₂', 1.0]], [105.14], [['DEA', 1]], [1188]]]

# multiply each groups weight by the last number in the list and replace the original values with the new ones
print(starting_materials)
for A in starting_materials:
    for B in A[0]:
        print(B[1], A[3][0])
        B[1] = B[1] * A[3][0]
print(starting_materials)




