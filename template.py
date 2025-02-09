import json



########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################

class DSU():
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0]*n
        self.n = n
    
    def find(self, u):
        if self.parent[u]==u:
            return u
        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        u = self.find(u)
        v = self.find(v)
        if u==v:
            return
        if self.rank[u]<self.rank[v]:
            u, v = v, u
        self.parent[v] = u
        if self.rank[u]==self.rank[v]:
            self.rank[u]+=1

class Graph():
    def __init__(self, n):
        self.adj = [set() for _ in range(n)]
        self.n = n
        self.vertex_to_cliques_mapping = [set() for _ in range(n)]
        self.cliques_to_vertex_mapping = []
        self.ordering = []
        self.orig_state = None
    
    def store_graph_state(self):
        self.orig_state = ([set([i for i in s]) for s in self.adj], [set([i for i in s]) for s in self.vertex_to_cliques_mapping], [set([i for i in s]) for s in self.cliques_to_vertex_mapping])

    def restore_state(self):
        self.adj=[set([i for i in s]) for s in self.orig_state[0]]
        self.vertex_to_cliques_mapping=[set([i for i in s]) for s in self.orig_state[1]]
        self.cliques_to_vertex_mapping=[set([i for i in s]) for s in self.orig_state[2]]

    def add_clique(self, clique): # clique is a list of vertices, 0-indexed
        for u in clique:
            self.vertex_to_cliques_mapping[u].add(len(self.cliques_to_vertex_mapping))
            for v in clique:
                if u==v:
                    continue
                self.adj[u].add(v)
                self.adj[v].add(u)
        self.cliques_to_vertex_mapping.append(set(clique))

    def delete_clique(self, clique_num):    
        for u in self.cliques_to_vertex_mapping[clique_num]:
            self.vertex_to_cliques_mapping[u].remove(clique_num)
        self.cliques_to_vertex_mapping[clique_num].clear()

    def min_neighbours_heuristic(self):
        ### TODO Make sure you deep copy before making any move. -> Nvm, this has been dealt with. I have implemented a save and restore functionality
        num_neighbours = [len(neighbours) for neighbours in self.adj]
        i=self.n
        simplicial = [i for i in range(self.n) if len(self.vertex_to_cliques_mapping[i])==1 and num_neighbours[i]>0]
        ordering = [i for i in range(self.n) if num_neighbours[i]==0]
        while i>0:
            while simplicial:
                node = simplicial.pop() # The node is in only one clique
                ordering.append(node)
                num_neighbours[node]=0
                simplicial += [i for i in range(self.n) if len(self.vertex_to_cliques_mapping[i])==1 and i not in simplicial and num_neighbours[i]>0]
                i-=1
                if (len(self.vertex_to_cliques_mapping[node])==0):
                    continue
                clique = self.vertex_to_cliques_mapping[node].pop()
                self.cliques_to_vertex_mapping[clique].remove(node)
                if len(self.cliques_to_vertex_mapping[clique])==1:
                    other_node_in_clique = self.cliques_to_vertex_mapping[clique].pop() 
                    self.vertex_to_cliques_mapping[other_node_in_clique].remove(clique)
                    if len(self.vertex_to_cliques_mapping[other_node_in_clique])==1:
                        simplicial.append(other_node_in_clique)
                for neighbour in self.adj[node]:
                    if num_neighbours[neighbour]==0:
                        continue
                    num_neighbours[neighbour]-=1
            if i==0:
                break
            element_with_min_neighbours = min([(num_neighbours[i], i) for i in range(self.n) if num_neighbours[i]>0])[1]
            for neighbour1 in self.adj[element_with_min_neighbours]:
                if num_neighbours[neighbour1]==0:
                    continue 
                for neighbour2 in self.adj[element_with_min_neighbours]:
                    if num_neighbours[neighbour2]==0:
                        continue
                    if neighbour1==neighbour2:
                        continue
                    if neighbour2 not in self.adj[neighbour1]:
                        self.adj[neighbour1].add(neighbour2)
                        self.adj[neighbour2].add(neighbour1)
                        num_neighbours[neighbour1]+=1
                        num_neighbours[neighbour2]+=1
            new_clique = [n for n in self.adj[element_with_min_neighbours] if num_neighbours[n] > 0]
            new_clique_copy = set(new_clique.copy())
            new_clique_copy.add(element_with_min_neighbours)
            # ordering.append(element_with_min_neighbours)
            for clique_num in range(len(self.cliques_to_vertex_mapping)):
                if element_with_min_neighbours in self.cliques_to_vertex_mapping[clique_num]:
                    self.delete_clique(clique_num)
            # Expand new_clique with all nodes that are now in the same clique
            # For any clique now majorized by the new large clique, delete it
            for node in range(self.n):
                if not node==element_with_min_neighbours and all([parent in self.adj[node] for parent in new_clique]):
                    new_clique.append(node)
            for clique_num in range(len(self.cliques_to_vertex_mapping)):
                if all([node in new_clique for node in self.cliques_to_vertex_mapping[clique_num]]):
                    self.delete_clique(clique_num)
                elif element_with_min_neighbours in self.cliques_to_vertex_mapping[clique_num]:
                    self.delete_clique(clique_num)
            self.add_clique(new_clique)
            simplicial.append(element_with_min_neighbours)
        print(ordering)
        self.ordering = ordering

    def min_fill_heuristic(self):
        num_neighbours = [len(neighbours) for neighbours in self.adj]
        num_fill = [sum([u!=v and u not in self.adj[v] for u in self.adj[i] for v in self.adj[i]])//2 for i in range(self.n)]
        i=self.n
        simplicial = [i for i in range(self.n) if len(self.vertex_to_cliques_mapping[i])==1 and num_neighbours[i]>0]
        ordering = [i for i in range(self.n) if num_neighbours[i]==0]
        while i>0:
            while simplicial:
                node = simplicial.pop() # The node is in only one clique
                ordering.append(node)
                num_neighbours[node]=0
                simplicial += [i for i in range(self.n) if len(self.vertex_to_cliques_mapping[i])==1 and i not in simplicial and num_neighbours[i]>0]
                i-=1
                if (len(self.vertex_to_cliques_mapping[node])==0):
                    continue
                clique = self.vertex_to_cliques_mapping[node].pop()
                self.cliques_to_vertex_mapping[clique].remove(node)
                if len(self.cliques_to_vertex_mapping[clique])==1:
                    other_node_in_clique = self.cliques_to_vertex_mapping[clique].pop() 
                    self.vertex_to_cliques_mapping[other_node_in_clique].remove(clique)
                    if len(self.vertex_to_cliques_mapping[other_node_in_clique])==1:
                        simplicial.append(other_node_in_clique)
                for neighbour in self.adj[node]:
                    if num_neighbours[neighbour]==0:
                        continue
                    for possible_immorality in self.adj[neighbour]:
                        if num_neighbours[possible_immorality]==0:
                            continue
                        if possible_immorality==node:
                            continue
                        if possible_immorality not in self.adj[node]:
                            num_fill[neighbour]-=1    
                    num_neighbours[neighbour]-=1
            if i==0:
                break
            # print([i for i in range(self.n) if num_neighbours[i]>0])
            element_with_min_neighbours = min([(num_fill[i], i) for i in range(self.n) if num_neighbours[i]>0])[1]
            for neighbour1 in self.adj[element_with_min_neighbours]:
                if num_neighbours[neighbour1]==0:
                    continue 
                for neighbour2 in self.adj[element_with_min_neighbours]:
                    if num_neighbours[neighbour2]==0:
                        continue
                    if neighbour1==neighbour2:
                        continue
                    if neighbour2 not in self.adj[neighbour1]:
                        self.adj[neighbour1].add(neighbour2)
                        self.adj[neighbour2].add(neighbour1)
                        num_neighbours[neighbour1]+=1
                        num_neighbours[neighbour2]+=1
            new_clique = [n for n in self.adj[element_with_min_neighbours] if num_neighbours[n] > 0]
            new_clique_copy = set(new_clique.copy())
            new_clique_copy.add(element_with_min_neighbours)
            # ordering.append(element_with_min_neighbours)
            for clique_num in range(len(self.cliques_to_vertex_mapping)):
                if element_with_min_neighbours in self.cliques_to_vertex_mapping[clique_num]:
                    self.delete_clique(clique_num)
            # Expand new_clique with all nodes that are now in the same clique
            # For any clique now majorized by the new large clique, delete it
            for node in range(self.n):
                if not node==element_with_min_neighbours and all([parent in self.adj[node] for parent in new_clique]):
                    new_clique.append(node)
            for clique_num in range(len(self.cliques_to_vertex_mapping)):
                if all([node in new_clique for node in self.cliques_to_vertex_mapping[clique_num]]):
                    self.delete_clique(clique_num)
                elif element_with_min_neighbours in self.cliques_to_vertex_mapping[clique_num]:
                    self.delete_clique(clique_num)
            self.add_clique(new_clique)
            simplicial.append(element_with_min_neighbours)
        print(ordering)
        self.ordering = ordering

class JunctionTree():
    def __init__(self, adj, simplicial_ordering):
        self.n = len(simplicial_ordering)
        self.cliques = []
        excluded = set()
        self.sep_sets = []
        for i in range(self.n):
            self.cliques.append([simplicial_ordering[i]])
            self.cliques[-1]+=[a for a in adj[simplicial_ordering[i]] if a not in excluded]
            self.cliques[-1]=clique=set(self.cliques[-1])
            self.sep_sets.append([])
            for j in range(len(self.cliques)-1):
                sep_set = clique & self.cliques[j]
                self.sep_sets[-1].append(-len(sep_set))
                if len(sep_set)==len(clique):
                    self.cliques.pop()
                    self.sep_sets.pop()
                    break
            excluded.add(simplicial_ordering[i])
        self.get_junction_tree()
    
    def get_junction_tree(self):
        # I also want to store a parent relation. The junction tree algorithm can work in two passes then.
        # Since my simplicial ordering should select leaf nodes first, then my parent must be the greaer index out of (i, j) when i-j is an edge
        self.dsu = DSU(self.n)
        self.sep_sets_list = [(self.sep_sets[i][j], i, j) for i in range(len(self.sep_sets)) for j in range(len(self.sep_sets[i]))]
        self.sep_sets_list.sort()
        final_edges = []
        for edge in self.sep_sets_list:
            if self.dsu.find(edge[1])!=self.dsu.find(edge[2]):
                final_edges.append((self.cliques[edge[1]] & self.cliques[edge[2]], self.cliques[edge[1]], self.cliques[edge[2]]))
                self.dsu.union(edge[1], edge[2])
        # print(final_edges)

class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        
        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.
        
        Refer to the sample test case for the structure of the input data.
        """
        self.variables_count = data['VariablesCount']
        self.k = data['k value (in top k)']
        self.graph = Graph(self.variables_count)
        for clique in data['Cliques and Potentials']:
            self.graph.add_clique(clique['cliques'])
        self.graph.store_graph_state()
        self.graph.min_neighbours_heuristic()
        self.junction_tree = JunctionTree(self.graph.adj, self.graph.ordering)
        self.graph.restore_state()
        self.graph.min_fill_heuristic()
        self.junction_tree = JunctionTree(self.graph.adj, self.graph.ordering)
        self.graph.restore_state()
        ### TODO This is a bad method of saving and restoring, which I am using for now. Improve this later.

    def triangulate_and_get_cliques(self):
        """
        Triangulate the undirected graph and extract the maximal cliques.
        
        What to do here:
        ----------------
        - Implement the triangulation algorithm to make the graph chordal.
        - Extract the maximal cliques from the triangulated graph.
        - Store the cliques for later use in junction tree creation.

        Refer to the problem statement for details on triangulation and clique extraction.
        """
        pass

    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.
        
        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.

        Refer to the problem statement for details on junction tree construction.
        """
        pass

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        
        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.
        
        Refer to the sample test case for how potentials are associated with cliques.
        """
        pass

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.
        
        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.
        
        Refer to the problem statement for details on computing the partition function.
        """
        pass

    def compute_marginals(self):
        """
        Compute the marginal probabilities for all variables in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to compute the marginal probabilities for each variable.
        - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.
        
        Refer to the sample test case for the expected format of the marginals.
        """
        pass

    def compute_top_k(self):
        """
        Compute the top-k most probable assignments in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.
        
        Refer to the sample test case for the expected format of the top-k assignments.
        """
        pass



########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    # evaluator = Get_Input_and_Check_Output('Sample_Testcase.json')
    evaluator = Get_Input_and_Check_Output('Generated_Testcase.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')