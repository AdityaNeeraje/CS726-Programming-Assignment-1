import json

i=0

class Clique:
    def __init__(self, vertices):
        self.size = len(vertices)
        self.vertices = vertices
        self.potentials = [0 for i in range (2**self.size)]

class Testcase:
    def __init__(self, n, cliques):
        global i
        i+=1
        self.TestCaseNumber = i
        self.n = n
        self.cliques = [Clique(clique) for clique in cliques]
        self.map = {"TestCaseNumber": self.TestCaseNumber, "Input": {"TestCaseNumber": self.TestCaseNumber, "VariablesCount": self.n, "Potentials_count": len(self.cliques),
                                                                     "Cliques and Potentials": [
                                                                         {"clique_size": clique.size, "cliques": clique.vertices, "potentials": clique.potentials}  for clique in self.cliques
                                                                     ],
                                                                    "k value (in top k)": 2,
                                                                     }
                                                                     }
        self.adj = [set() for _ in range(n)]
        self.n = n

data=[Testcase(3, [[0, 1], [1, 2]]), Testcase(7, [[0,1], [1, 2], [1, 3], [0, 4], [4, 5, 6]])]
data.append(Testcase(8, [[0,1], [1, 2], [1, 3], [0, 4], [4, 5], [5, 7], [6, 7], [4, 6]]))
data.append(Testcase(10, [[0,1], [1,2], [2,3], [3,0], [3, 4], [4,5],[4,6],[6,7],[6,8],[8,9],[8,2]]))
with open("Generated_Testcase.json", "w") as f:
    json.dump([d.map for d in data], f, indent=4)