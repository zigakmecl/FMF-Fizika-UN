import math


def dijkstra(graf, s):
    def povezava(x, y):
        return x.get(y, math.inf)

    obiskani = []

    # print(graf)
    razdalja = dict()
    for i in graf.keys():
        razdalja[i] = math.inf
    razdalja[s] = 0
    # obiskani.append(s)
    print((graf[s]))

    # for i in razdalja.keys():
    #     for j, k in graf[i].items():
    #         razdalja[j] = k
    #         # obiskani.append(j)
    razdalja = dict(sorted(razdalja.items(), key=lambda x: x[1]))
    while len(obiskani) < len(razdalja):
        for i in razdalja.keys():
            if i not in obiskani:
                for j in razdalja.keys():
                    do_y_skozi_x = razdalja[i] + povezava(graf[i], j)
                    if do_y_skozi_x < razdalja[j]:
                        razdalja[j] = do_y_skozi_x
                obiskani.append(i)
                break
        razdalja = dict(sorted(razdalja.items(), key=lambda x: x[1]))
    return razdalja


graf_wiki = {
    "1": {"2": 7, "3": 9, "6": 14},
    "2": {"1": 7, "3": 10, "4": 15},
    "3": {"1": 9, "2": 10, "4": 11, "6": 2},
    "4": {"2": 15, "3": 11, "5": 6},
    "5": {"4": 6, "6": 9},
    "6": {"1": 14, "3": 2, "5": 9},
}

graf_predavanja = {
    "Ljubljana": {"Maribor": 127, "Kranj": 26},
    "Maribor": {"Ljubljana": 127},
    "Koper": {"Ljubljana": 97, "Kranj": 105},
    "Kranj": {"Maribor": 140, "Koper": 105},
}

print(dijkstra(graf_wiki, "5"))
# print(dijkstra(graf_predavanja, "Ljubljana"))
