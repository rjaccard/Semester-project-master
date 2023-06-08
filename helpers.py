import numpy as np
from collections import Counter

colors=['#636EFA',
 '#EF553B',
 '#00CC96',
 '#AB63FA',
 '#FFA15A',
 '#19D3F3',
 '#FF6692',
 '#B6E880',
 '#FF97FF',
 '#FECB52']

# ##############################################
# Helpers functions
# ##############################################

def is_resolving_set(G, nodes_in_subset, length):
    """Given a graph and the matrix with all shortest paths, 
    test if a set of node resolve the graph

    Args:
        G (Graph): A graph
        nodes_in_subset (set): A set of nodes
        length (dict): Dictionary with all shortest path

    Returns:
        bool: true if the set of nodes resolves the graph and false otherwise
    """

    dist = {}

    # For each node in G, compute the shortest path lengths to the nodes in nodes_in_subset
    for node in G.nodes():
        distances_subset = {}
        for n in nodes_in_subset:
            distances_subset[n] = length[node][n]
        dist[node] = tuple(distances_subset.values())

    # Check if the vector with all the shortest paths to nodes nodes_in_subset is different for each node in G
    # If it is, then the set of nodes nodes_in_subset resolves the graph G, otherwise it does not
    res = len((set(list(dist.values())))) == G.number_of_nodes()   

    return res



def perc_resolving_set(G, nodes_in_subset, length):
    """Given a graph and the matrix with all shortest paths, 
    test if a set of node resolve the graph

    Args:
        G (Graph): A graph
        nodes_in_subset (set): A set of nodes
        length (dict): Dictionary with all shortest path

    Returns:
        bool: true if the set of nodes resolves the graph and false otherwise
    """

    dist = {}

    # For each node in G, compute the shortest path lengths to the nodes in nodes_in_subset
    for node in G.nodes():
        distances_subset = {}
        for n in nodes_in_subset:
            distances_subset[n] = length[node][n]
        dist[node] = tuple(distances_subset.values())

    # Check if the vector with all the shortest paths to nodes nodes_in_subset is different for each node in G
    # If it is, then the set of nodes nodes_in_subset resolves the graph G, otherwise it does not
    res = len((set(list(dist.values())))) / G.number_of_nodes()   

    return res

def set_resolved(G, nodes_in_subset, length):
    """Given a graph and the matrix with all shortest paths, 
    output the set of nodes that are resolved by the given subset

    Args:
        G (Graph): A graph
        nodes_in_subset (set): A set of nodes
        length (dict): Dictionary with all shortest path

    Returns:
        bool: true if the set of nodes resolves the graph and false otherwise
    """

    dist = {}

    # For each node in G, compute the shortest path lengths to the nodes in nodes_in_subset
    for node in G.nodes():
        distances_subset = {}
        for n in nodes_in_subset:
            distances_subset[n] = length[node][n]
        dist[node] = tuple(distances_subset.values())  

    return get_unique_keys(dist)

def get_nodes_with_diff_neighbors(G, nodes):
    output = []
    while len(nodes) != 0:
        element = nodes[0]
        nodes.remove(element)
        output.append(element)
        neighbors = list(G.neighbors(element))
        for neighbor in neighbors:
            if neighbor in nodes:
                nodes.remove(neighbor)
            for el in nodes:
                if neighbor in list(G.neighbors(el)):
                    nodes.remove(el)
                
    return output

def closest_to_index(lst, to=0.5):
        
    """Find the index of the element in lst that is closest to the value 'to'
    
    Args:
        lst (list): A list of numbers
        to (float, optional): The target value to find the closest element to (default: 0.5)
    
    Returns:
        int: The index of the element in lst that is closest to the value 'to'
    """
        
    arr = np.array(lst)
    idx = np.abs(arr - to).argmin()

    return idx

def diag_mat(n, p, q):

    # Define the diagonal value and the rest of the matrix value
    diag_value = p
    rest_value = q

    # Create an empty matrix of zeros with the specified dimensions
    matrix = np.zeros((n, n))

    # Fill the diagonal with the diagonal value
    np.fill_diagonal(matrix, diag_value)

    # Fill the rest of the matrix with the rest value
    matrix[matrix == 0] = rest_value

    return matrix

def get_unique_keys(dictionary):
    value_counter = Counter(dictionary.values())
    unique_keys = [key for key, value in dictionary.items() if value_counter[value] == 1]
    return unique_keys


# ##############################################
# Theoretical values
# ##############################################

def tvalue_case_1_2_p_const(n, p): #f1
    """Theoretical value proven in "Metric dimension for random graphs"
    from Bela Bollobasm, case i and ii with p constant

    Args:
        n (int): number of nodes
        p (float): probability of two nodes being connected

    Returns:
        float: theoretical value
    """
    return 2*np.log(n) / np.log(1/(p**2 + (1-p)**2))

def tvalue_case_1_2_p_o1(n, p, i):
    """Theoretical value proven in "Metric dimension for random graphs"
    from Bela Bollobasm, case i and ii with p going to zero

    Args:
        n (int): number of nodes
        p (float): probability of two nodes being connected

    Returns:
        float: theoretical value
    """
    n = np.array(n)
    c = (np.power(p*(n-1), i+1)) / n
    value = 2*np.log(n) / np.log(1/(np.power(np.exp(-c),2) + (1-np.power(np.exp(-c),2))))
    return value

def tvalue_case_3(n, p, i):
    """Theoretical value proven in "Metric dimension for random graphs"
    from Bela Bollobasm, case iii

    Args:
        n (int): number of nodes
        p (float): probability of two nodes being connected

    Returns:
        float: theoretical value
    """
    n = np.array(n)
    c = (np.power(p*(n-1), i+1)) / n
    value = 1/(((p*(n-1))**i/n)+np.exp(-c))
    return value


def tvalue_odor(n, p):
    """Theoretical value proven in "The Role of Adaptivity in Source 
    Identification with Time Queries" from Gergely ODOR when p constant

    Args:
        n (int): number of nodes
        p (float): probability of two nodes being connected

    Returns:
        float: theoretical value
    """
    return np.log(n*(1-np.sqrt(p**2 + (1-p)**2))) / np.log(1/np.sqrt(p**2 + (1-p)**2))


def cvalue_communities(n, p, q, k): #f12c
    """Conjecture value for the metric dimension of SBM graphs with p the intra-cluster edge probability 
    and q the inter-cluster probability.
    """
    gamma = (np.sqrt(p**2 + (1-p)**2)**(1/k)) * (np.sqrt(q**2 + (1-q)**2)**((k-1)/k))
    return np.log(n) / np.log(1/gamma)