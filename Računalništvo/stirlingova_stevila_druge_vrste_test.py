inf = float("inf")


def floyd_warshall(d):
    f = [[x for x in vrstica] for vrstica in d]
    for i in range(len(f)):
        for j in range(len(f)):
            for k in range(len(f)):
                if f[i][j] > f[i][k] + f[k][j]:
                    f[i][j] = f[i][k] + f[k][j]
    return f


def najkrajsa_pot(d, f, ii, jj):
    if d[ii][jj] == f[ii][jj]:
        return [i, j]
    elif f[ii][jj] == inf:
        return None
    else:
        h = [[[b, a] for a in range(len(d))] for b in range(len(d))]
        f2 = d.copy()
        for i in range(len(f)):
            for j in range(len(f)):
                for k in range(len(f)):
                    if f2[i][j] > f2[i][k] + f2[k][j]:
                        f2[i][j] = f2[i][k] + f2[k][j]
                        h[i][j] = [a for a in h[i][k]]
                        h[i][j].extend(h[k][j][1:])
                    elif h[i][j] == []:
                        if i != j:
                            h[i][j] = [i, j]
                        else:
                            h[i][j] = [i]
        print(h)
        return h[ii][jj]


d = [
    [0, 2, inf, inf, 1],
    [2, 0, 1, inf, inf],
    [inf, 1, 0, 1, inf],
    [inf, inf, 1, 0, 2],
    [1, inf, inf, 2, 0],
]
f = floyd_warshall(d)
print(najkrajsa_pot(d, f, 0, 3))
