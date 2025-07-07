def find_edges_with_vertex(edges, vertex):
    """
    Find up to two edges from a list that contain a given vertex.
    
    Parameters:
    -----------
    edges : list of tuples
        List of edges, where each edge is a tuple (i, j) representing vertices
    vertex : int
        The vertex to search for
        
    Returns:
    --------
    list of tuples
        List of up to 2 edges that contain the given vertex
    """
    result = []
    for edge in edges:
        if vertex in edge:
            result.append(edge)
            if len(result) == 2:  # Stop after finding 2 edges
                break
    return result

# Example usage
if __name__ == "__main__":
    # Test with your example
    edges_A = [(0, 1), (0, 2), (1, 3)]
    
    print("Two edges containing vertex 0:")
    print(find_edges_with_vertex(edges_A, 0))  # Should return [(0, 1), (0, 2)]
    
    print("Two edges containing vertex 1:")
    print(find_edges_with_vertex(edges_A, 1))  # Should return [(0, 1), (1, 3)]
    
    print("Two edges containing vertex 2:")
    print(find_edges_with_vertex(edges_A, 2))  # Should return [(0, 2)]
    
    print("Two edges containing vertex 3:")
    print(find_edges_with_vertex(edges_A, 3))  # Should return [(1, 3)] 