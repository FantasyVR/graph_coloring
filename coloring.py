import numpy as np
from scipy.sparse import csr_matrix


def compute_max_degree(edges):
    adj_matrix = np.zeros((len(edges), len(edges)), dtype=np.int32)
    for i in range(len(edges) - 1):
        for j in range(i + 1, len(edges)):
            e1, e2 = edges[i], edges[j]
            if e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[
                    1] == e2[1]:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
    non_zeros = np.count_nonzero(adj_matrix, axis=1)
    max_degree = np.max(non_zeros)
    min_degree = np.min(non_zeros)
    return adj_matrix, max_degree, min_degree


"""
Reference: Fig.2 of A PARALLEL GRAPH COLORING HEURISTICR, Mark T. Jones and Paul E. Plassmann, 1993.
"""


def monte_carlo_coloring(e):
    adj_matrix, max_degree, min_degree = compute_max_degree(e)
    print(f"max degree: {max_degree}")
    adj_csr = csr_matrix(adj_matrix)
    row_offsets, col_indices = adj_csr.indptr, adj_csr.indices
    V = [i for i in range(len(e))]  # uncolored vertices
    color = 0
    c_v = [-1] * len(e)  # clolors for each vertex
    while len(V) > 0:
        # Step 1: Choose an independent set I from V: Monte Carlo Method
        I = []
        rho = [0] * len(V)
        # Step 1.1: For each vertex v in V determine a distinct, random number rho(v)
        for i in range(len(V)):
            pro = np.random.randint(0, 10 * len(V))
            while pro in rho:
                pro = np.random.randint(0, 10 * len(V))
            rho[i] = pro
        # Step 1.2: v in I, if and only if: rho(v) > rho(w) for any w in adj (v)
        for i in range(len(V)):
            e_idx = V[i]
            s_col, e_col = row_offsets[e_idx], row_offsets[e_idx + 1]
            neighbor_idx = col_indices[s_col:e_col]
            select = True
            for neighbor in neighbor_idx:
                if neighbor in V:
                    if rho[i] < rho[V.index(neighbor)]:
                        select = False
                        break
            if select:
                I.append(e_idx)
        # color I in parallel
        for v in I:
            c_v[v] = color
        color += 1
        # remove I from V
        for v in I:
            V.remove(v)
    return c_v


"""
Reference: A Simple Parallel Algorithm for the Maximal Independent Set Problem. Michael Luby.
https://en.wikipedia.org/wiki/Maximal_independent_set
"""


def luby_coloring(e):
    adj_matrix, max_degree, min_degree = compute_max_degree(e)
    print(f"max degree: {max_degree}")
    adj_csr = csr_matrix(adj_matrix)
    row_offsets, col_indices = adj_csr.indptr, adj_csr.indices
    V_global = [i for i in range(len(e))]  # uncolored vertices
    color = 0
    c_v = [-1] * len(e)  # clolors for each vertex
    while (len(V_global) > 0):
        I = []
        V = V_global.copy()
        while len(V) > 0:
            S = []
            pro = np.random.uniform(0, 1, len(V))
            for i in range(len(V)):
                v = V[i]
                s_col, e_col = row_offsets[v], row_offsets[v + 1]
                dv = e_col - s_col  # degree of vertex v
                if pro[i] < 1.0 / (2.0 *
                                   dv):  # select v with probability: 1/(2dv)
                    S.append(v)
            for row in range(len(row_offsets) - 1):
                s_col, e_col = row_offsets[row], row_offsets[row + 1]
                dv_row = e_col - s_col
                for col in col_indices[s_col:e_col]:
                    if row in S and col in S:
                        dv_col = row_offsets[col + 1] - row_offsets[col]
                        if dv_row < dv_col:
                            S.remove(row)
                        else:
                            S.remove(col)
            I = I + S
            # find all neighbors of S
            for v in S:
                V.remove(v)
                s_col, e_col = row_offsets[v], row_offsets[v + 1]
                neighbor_idx = col_indices[s_col:e_col]
                for neighbor in neighbor_idx:
                    if neighbor in V:
                        V.remove(neighbor)
        # color I in parallel
        for v in I:
            c_v[v] = color
        color += 1
        # remove I from V_global
        for v in I:
            V_global.remove(v)
    return c_v


def luby_MIS(e):
    adj_matrix, max_degree, min_degree = compute_max_degree(e)
    print(f"max degree: {max_degree}")
    adj_csr = csr_matrix(adj_matrix)
    row_offsets, col_indices = adj_csr.indptr, adj_csr.indices
    V = [i for i in range(len(e))]  # uncolored vertices
    I = []
    while len(V) > 0:
        S = []
        pro = np.random.uniform(0, 1, len(V))
        for i in range(len(V)):
            v = V[i]
            s_col, e_col = row_offsets[v], row_offsets[v + 1]
            dv = e_col - s_col  # degree of vertex v
            if pro[i] < 1.0 / (2.0 * dv):  # select v with probability: 1/(2dv)
                S.append(v)
        for row in range(len(row_offsets) - 1):
            s_col, e_col = row_offsets[row], row_offsets[row + 1]
            dv_row = e_col - s_col
            for col in col_indices[s_col:e_col]:
                if row in S and col in S:
                    dv_col = row_offsets[col + 1] - row_offsets[col]
                    if dv_row < dv_col:
                        S.remove(row)
                    else:
                        S.remove(col)
        I = I + S
        # find all neighbors of S
        for v in S:
            V.remove(v)
            s_col, e_col = row_offsets[v], row_offsets[v + 1]
            neighbor_idx = col_indices[s_col:e_col]
            for neighbor in neighbor_idx:
                if neighbor in V:
                    V.remove(neighbor)
    return I


"""
Sec. 7: Delta+1VC Problem. 
Reference: A Simple Parallel Algorithm for the Maximal Independent Set Problem. Michael Luby.
"""


def sequential_greedy_coloring_heuristic(e):
    adj_matrix, max_degree, min_degree = compute_max_degree(e)
    print(f"max degree: {max_degree}")
    adj_csr = csr_matrix(adj_matrix)
    row_offsets, col_indices = adj_csr.indptr, adj_csr.indices
    color_sets = [i for i in range(max_degree + 1)]
    e_color = [-1] * len(e)  # init edge color as -1
    # serial coloring method
    for i in range(len(e)):
        if e_color[i] != -1:
            continue
        # get all neighbor's color
        s_col, e_col = row_offsets[i], row_offsets[i + 1]
        neighbor_idx = col_indices[s_col:e_col]
        neighbor_color = []
        for neighbor in neighbor_idx:
            if e_color[neighbor] != -1:
                neighbor_color.append(e_color[neighbor])
        for k in range(len(color_sets)):
            if k not in neighbor_color:
                e_color[i] = k
                break
    return e_color


"""
Reference: Vivace: a Practical Gauss-Seidel Method for Stable Soft Body Dynamics. Marco Fratarcangeli et al.
"""


def vivace_coloring(e):
    adj_matrix, max_degree, min_degree = compute_max_degree(e)
    print(f"max degree: {max_degree}")
    adj_csr = csr_matrix(adj_matrix)
    row_offsets, col_indices = adj_csr.indptr, adj_csr.indices
    # Initialization
    U = [i for i in range(len(e))]  # uncolored vertices
    delta_V = [0] * len(e)  # the degree of each vertex
    for i in range(len(e)):
        delta_V[i] = row_offsets[i + 1] - row_offsets[i]
    s = 1  # smallest degree
    P_v = [0] * len(e)  # the platte of colors for each vertex
    max_color_idx = [0] * len(e)
    for i in range(len(e)):
        num_colors_in_platte = int(delta_V[i] / s) + 1
        P_v[i] = [i for i in range(num_colors_in_platte)]
        max_color_idx[i] = num_colors_in_platte - 1

    c_v = [-1] * len(e)  # clolors for each vertex
    while len(U) > 0:
        # Tentative coloring: parallel
        for v in U:
            num_colors_in_platte = len(P_v[v])
            color_idx = np.random.randint(0, num_colors_in_platte)
            c_v[v] = P_v[v][color_idx]  # randomly pick a color from the platte
        # Conflict resolution: parallel
        I = []
        for v in U:
            s_col, e_col = row_offsets[v], row_offsets[v + 1]
            neighbor_idx = col_indices[s_col:e_col]
            S = []  # colors of all the neighbors of v
            for neighbor in neighbor_idx:
                S.append(c_v[neighbor])
            if c_v[v] not in S:
                I.append(v)
                for neighbor in neighbor_idx:
                    if c_v[v] in P_v[neighbor]:
                        P_v[neighbor].remove(c_v[v])
        for v in I:
            U.remove(v)

        # Feed the hungry: parallel
        for v in U:
            if len(P_v[v]) == 0:
                max_color_idx[v] += 1
                P_v[v].append(max_color_idx[v])
    return c_v


"""
Reference: A PARALLEL GRAPH COLORING HEURISTICR, Mark T. Jones and Paul E. Plassmann, 1993.
"""


def jones_plassmann_coloring(e):
    pass


"""
Section 2: Algorithm 1, 2, 3
Reference: Greed is Good: Parallel Algorithms for Bipartite-Graph Partial Coloring on
Multicore Architectures
"""


def greedy_coloring(e):
    adj_matrix, max_degree, min_degree = compute_max_degree(e)
    print(f"max degree: {max_degree}")
    adj_csr = csr_matrix(adj_matrix)
    row_offsets, col_indices = adj_csr.indptr, adj_csr.indices
    c_v = [-1] * len(e)  # clolors for each vertex
    W = [i for i in range(len(e))]  # uncolored vertices
    while len(W) > 0:
        # Color work queue
        for w in W:  # in parallel
            F = []
            s_col, e_col = row_offsets[w], row_offsets[w + 1]
            neighbor_idx = col_indices[s_col:e_col]
            for neighbor in neighbor_idx:
                if c_v[neighbor] != -1:
                    F.append(c_v[neighbor])
            col = 0
            while col in F:  # first-fit coloring policy
                col += 1
            c_v[w] = col
        # Remove conflicts
        W_next = []
        for w in W:  # in parallel
            s_col, e_col = row_offsets[w], row_offsets[w + 1]
            neighbor_idx = col_indices[s_col:e_col]
            for neighbor in neighbor_idx:
                if c_v[neighbor] == c_v[w] and w > neighbor:
                    W_next.append(w)
                    break
        W = W_next
    return c_v


"""
Reference: https://en.wikipedia.org/wiki/Maximal_independent_set#Random-priority_parallel_algorithm
"""


def random_priority_parallel_alg(e):
    adj_matrix, max_degree, min_degree = compute_max_degree(e)
    print(f"max degree: {max_degree}")
    adj_csr = csr_matrix(adj_matrix)
    row_offsets, col_indices = adj_csr.indptr, adj_csr.indices
    W_global = [i for i in range(len(e))]  # uncolored vertices
    color = 0
    c_v = [-1] * len(e)  # clolors for each vertex
    while len(W_global) > 0:
        W = W_global.copy()  # uncolored vertices
        I = []
        while len(W) > 0:
            rv = np.random.uniform(0, 1, len(W))
            for i in range(len(W)):
                v = W[i]
                s_col, e_col = row_offsets[v], row_offsets[v + 1]
                neighbor_idx = col_indices[s_col:e_col]
                flag = True
                for neighbor in neighbor_idx:
                    if neighbor in W:
                        idx = W.index(neighbor)
                        if rv[i] > rv[idx]:
                            flag = False
                            break
                if flag:
                    I.append(v)
            for v in I:
                s_col, e_col = row_offsets[v], row_offsets[v + 1]
                neighbor_idx = col_indices[s_col:e_col]
                for neighbor in neighbor_idx:
                    if neighbor in W:
                        W.remove(neighbor)
        # color I in parallel
        for v in I:
            c_v[v] = color
        color += 1
        # remove I from W_global in parallel
        for v in I:
            if v in W_global:
                W_global.remove(v)
    return c_v


def check_validity(e, c_v):
    for i in range(len(e) - 1):
        for j in range(i + 1, len(e)):
            if e[i][0] == e[j][0] or e[i][1] == e[j][1] or e[i][1] == e[j][
                    0] or e[i][0] == e[j][1]:
                if c_v[i] == c_v[j]:
                    print(f"{i} and {j} are in the same color")
                    return False
    return True


def graph_coloring(e, algorithm):
    func = [
        sequential_greedy_coloring_heuristic, vivace_coloring, greedy_coloring,
        monte_carlo_coloring, luby_coloring, check_validity
    ]
    c_v = func[algorithm](e)
    print("check validity:", check_validity(e, c_v))
    return c_v

if __name__ == "__main__":
    """
    0---3---6
    | \ | / |
    1---4---7
    | / | \ |
    2---5---8
    """
    edges = np.array([[0, 1], [1, 2], [0, 3], [0, 4], [1, 4], [2, 4], [2, 5],
                      [3, 4], [4, 5], [3, 6], [4, 6], [4, 7], [4, 8], [5, 8],
                      [6, 7], [7, 8]])
    """
    0---2
    |   |
    1---3
    """
    # edges = np.array([[0, 1], [1, 3], [3, 2], [2, 0]])
    e_color = sequential_greedy_coloring_heuristic(edges)
    print(f"sequential greedy coloring: {e_color}")

    e_color = monte_carlo_coloring(edges)
    print(f"monte carlo coloring: {e_color}")

    # MIS coloring
    e_color = luby_coloring(edges)
    print(f"luby coloring: {e_color}")

    # random platte coloring
    e_color = vivace_coloring(edges)
    print(f"vivace coloring: {e_color}")

    # Greedy coloring
    e_color = greedy_coloring(edges)
    print(f"greedy coloring: {e_color}")

    # Random priority parallel algorithm
    e_color = random_priority_parallel_alg(edges)
    print(f"random priority parallel algorithm: {e_color}")
