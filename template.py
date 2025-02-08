import json



########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################

class Graph():
    def __init__(self, n):
        self.adj = [set() for _ in range(n)]
        self.n = n
        self.vertex_to_cliques_mapping = [set() for _ in range(n)]
        self.cliques_to_vertex_mapping = []

    def add_clique(self, clique): # clique is a list of vertices, 0-indexed
        for u in clique:
            self.vertex_to_cliques_mapping[u].add(len(self.cliques_to_vertex_mapping))
            for v in clique:
                if u==v:
                    continue
                self.adj[u].add(v)
                self.adj[v].add(u)
        self.cliques_to_vertex_mapping.append(set(clique))

    def min_neighbours_heuristic(self):
        oo=1e9
        num_neighbours = [len(neighbours) for neighbours in self.adj]
        i=self.n
        simplicial = [i for i in range(self.n) if len(self.vertex_to_cliques_mapping[i])==1]
        ordering = []
        while i>0:
            while simplicial:
                node = simplicial.pop() # The node is in only one clique
                ordering.append(node)
                num_neighbours[node]=0
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
            new_clique = [n for n in self.adj[element_with_min_neighbours] if num_neighbours[n] > 0]
            new_clique += [intersection]
            self.add_clique([n for n in self.adj[element_with_min_neighbours] if num_neighbours[n] > 0])
            print([n for n in self.adj[element_with_min_neighbours] if num_neighbours[n] > 0])
            break
        print(ordering)

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
        self.graph.min_neighbours_heuristic()

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