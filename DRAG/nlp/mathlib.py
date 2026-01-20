import numpy as np

def max_profit_route(reward: np.ndarray, cost: np.ndarray, K: int) -> list:
    """
    Finds the maximum profit route with exactly K stops using dynamic programming.
    :param reward: List of rewards for each node.
    :param cost: 2D list of costs between nodes (cost[i][j] is the cost from node i to node j).
    :param K: Number of stops to make.
    :return: List of nodes in the route.
    """
    N = len(reward)  # Number of nodes
    
    # Initialize dp and path matrices
    dp = -np.inf * np.ones((N, K), dtype=float)  # Max profit table
    path = -np.ones((N, K), dtype=int)           # Path table
    
    # First stop: only the reward of each node
    dp[:, 0] = reward

    # Fill the dp table for stops 2 to K
    for k in range(1, K):  # Loop for each stop from 2 to K
        for i in range(N):  # For each node as a destination
            # Check all previous nodes j to come to node i
            for j in range(i):
                # Calculate the total profit if we come from node j to node i
                profit = dp[j, k - 1] + reward[i] - cost[j, i]
                # Update dp if we found a better profit
                if profit > dp[i, k]:
                    dp[i, k] = profit
                    path[i, k] = j  # Store the previous node
    
    # Find the node with the maximum profit at the final stop K
    end_node = np.argmax(dp[:, K - 1])
    
    # Backtrack to find the route
    route = [int(end_node)]
    for k in range(K - 1, 0, -1):
        end_node = path[end_node, k]
        route.append(int(end_node))

    route.reverse()  # Reverse the route to get the correct order
    return route

def overlap_sim_matrix(tf_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the keyword similarity matrix from the term frequency matrix using cosine similarity.
    :param tf_matrix: The term frequency matrix (num_terms x num_sentences).
    :return: The similarity matrix (num_sentences x num_sentences).
    """
    col_norms = np.clip(np.linalg.norm(tf_matrix, axis=0, keepdims=True), 1, None)
    sim_matrix = np.dot(tf_matrix.T, tf_matrix) / np.outer(col_norms, col_norms)
    return sim_matrix

def embed_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes the semantic similarity matrix from the embeddings using cosine similarity.
    :param embeddings: The embeddings matrix (num_samples x embedding_dim).
    :return: The similarity matrix (num_samples x num_samples).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    sim_matrix = np.clip(sim_matrix, 0, 1)  # Ensure values are between 0 and 1
    return sim_matrix

def markov_chain(transition_matrix: np.ndarray, v: np.ndarray = None, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Computes the steady state distribution of a Markov chain using power iteration.
    :param transition_matrix: The transition matrix of the Markov chain. Rows should sum to 1.
    :param v: Initial distribution. If None, uniform distribution is used.
    :param max_iter: Maximum number of iterations.
    :param tol: Tolerance for convergence.
    :return: Steady state distribution.
    """
    num_nodes = transition_matrix.shape[0]
    if v is None:
        v = np.ones(num_nodes) / num_nodes
    else:
        v = v
    for _ in range(max_iter):
        v_next = np.dot(transition_matrix.T, v)
        if np.linalg.norm(v - v_next) < tol:
            break
        v = v_next
    return v

def graph_reduction(graph: np.ndarray) -> np.ndarray:
    """
    Ranks the nodes of a graph based on their degree and returns a list of nodes.
    If the graph is a transitory matrix then the sum of the rows must be 1.
    Can be used as a quicker but less accurate alternative to the Markov chain.
    :param graph: The matrix of the graph.
    """
    return np.sum(graph, axis=0)