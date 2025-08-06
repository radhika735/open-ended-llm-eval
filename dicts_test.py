all = []
i = 0
while i < 5:
    iteration = {"num":i, "singles":[]}
    all.append(iteration)
    for j in ['a','b','c']:
        single = {"letter":j}
        iteration["singles"].append(single)
    i += 1
print(all)