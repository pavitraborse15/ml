def mp_neuron(x1, x2):
    net = x1 - x2
    return 1 if net >= 1 else 0

data = [(0,0), (0,1), (1,0), (1,1)]

print("X1 X2 Output")
for x1, x2 in data:
    print(x1, x2, mp_neuron(x1, x2))
    