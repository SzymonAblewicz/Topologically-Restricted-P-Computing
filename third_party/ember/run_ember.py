import networkx as nx
import dwave_networkx as dnx
from pssa_ember import PSSADefault 

def main():
    # 1. Create a logical source problem graph (a math problem for it to solve)
    source_graph = nx.complete_graph(8)
    
    # 2. Create a target hardware graph (simulating the D-Wave quantum computer)
    target_graph = dnx.chimera_graph(16)
    
    # 3. Load up the PSSA engine
    embedder = PSSADefault()
    
    # 4. Run the embedding
    print("Running PSSA...")
    result = embedder.embed(source_graph, target_graph, timeout=30.0)
    
    # 5. Output the results
    if result.get('status') == 'FAILURE':
        print("Failed to find an embedding:", result.get('error', 'Timeout or unknown error'))
    else:
        embedding = result.get('embedding', {})
        print(f"Successfully found an embedding in {result['time']:.2f} seconds!")
        for logical_node, physical_chain in embedding.items():
            print(f"Logical Node {logical_node} -> Physical Qubits {physical_chain}")

if __name__ == "__main__":
    main()