import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import pickle
from datasets import load_dataset
from tqdm import tqdm
import argparse

def load_embeddings_and_graph(reference_embed_path, skill_graph_path):
    """
    Load benchmark content embeddings and concept graph.
    
    Args:
        reference_embed_path (str): Path to benchmark content embeddings
        skill_graph_path (str): Path to concept graph pickle file
        
    Returns:
        tuple: (benchmark_embeddings, adjacency_matrix, node_info)
    """
    # Load benchmark content embeddings
    benchmark_content_embeds = np.load(reference_embed_path)
    
    # Load concept graph
    with open(skill_graph_path, "rb") as f:
        G_adj = pickle.load(f)
    
    # Convert graph to adjacency matrix
    adj_matrix_numpy = nx.to_numpy_array(G_adj)
    adj_matrix = torch.tensor(adj_matrix_numpy, dtype=torch.float32, device='cuda')
    
    # Extract node information
    node_kp_ls = []
    count_ls = []
    content_ls_ls = []
    
    for node, attrs in G_adj.nodes(data=True):
        content_ls_ls.append(attrs['list_indices'])
        node_kp_ls.append(node)
        count_ls.append(attrs['count'])
    
    return benchmark_content_embeds, adj_matrix, content_ls_ls

def compute_similarity_scores(test_embeddings, benchmark_embeddings, content_ls_ls, adj_matrix, chunk_size=10000):
    """
    Compute similarity scores between test data and benchmark data.
    
    Args:
        test_embeddings (np.ndarray): Test data embeddings
        benchmark_embeddings (np.ndarray): Benchmark data embeddings
        content_ls_ls (list): List of content indices for each node
        adj_matrix (torch.Tensor): Adjacency matrix of the concept graph
        chunk_size (int): Size of chunks for processing
        
    Returns:
        torch.Tensor: Final similarity scores
    """
    # Convert embeddings to tensors and move to GPU
    tensor_a = torch.from_numpy(test_embeddings).cuda()
    tensor_b = torch.from_numpy(benchmark_embeddings).cuda()
    
    pooling_score_ls = []
    
    # Process in chunks to manage memory
    for i in tqdm(range(0, tensor_a.size(0), chunk_size), desc="Processing chunks"):
        end = min(i + chunk_size, tensor_a.size(0))
        chunk = tensor_a[i:end]
        
        # Compute cosine similarities
        similarities_ls = []
        for j in range(chunk.shape[0]):
            cosine_sim = F.cosine_similarity(chunk[j], tensor_b, dim=1)
            similarities_ls.append(cosine_sim)
        
        similarities = torch.stack(similarities_ls)
        
        # Compute knowledge point similarities
        arrays = []
        for indices in content_ls_ls:
            min_values = torch.min(similarities[:, indices], dim=1).values
            arrays.append(min_values)
        
        kp_similarities = torch.stack(arrays, dim=1)
        kp_similarities = kp_similarities.to(dtype=torch.float32)
        
        # Apply adjacency matrix transformation
        final_matrix = torch.mm(kp_similarities, adj_matrix)
        final_matrix_tensor = torch.tensor(final_matrix, dtype=torch.float32, device='cuda')
        
        # Compute final scores
        final_score_tensor = final_matrix_tensor
        pooling_score_tensor = torch.sum(final_score_tensor, dim=1)
        pooling_score_ls.append(pooling_score_tensor)
    
    # Combine scores from all chunks
    pooling_score = torch.cat(pooling_score_ls, dim=0)
    return pooling_score

def main():
    """
    Main function to run the data selection process.
    """
    parser = argparse.ArgumentParser(description='Rank target data based on similarity scores')
    parser.add_argument('--reference_embed_path', type=str, 
                        default='./numina_100k_content_embed.npy',
                        help='Path to benchmark content embeddings')
    parser.add_argument('--skill_graph_path', type=str,
                        default="/mnt/jznas2/jiazheng.li/concept_graph/graphs/graph_numina_100k_qwen25_new_adj_1.pkl",
                        help='Path to skill graph pickle file')
    parser.add_argument('--target_embed_path', type=str,
                        default='./jiuzhang_embeddings.npy',
                        help='Path to test target data embeddings')
    parser.add_argument('--scores_output_path', type=str,
                        default='./score_jiuzhang_numina100k_newadj_1.pt',
                        help='Path to save similarity scores')
    parser.add_argument('--target_dataset_path', type=str,
                        default='./jiuzhang3-corpus-pt-cot.jsonl',
                        help='Path to input dataset')
    parser.add_argument('--ranked_dataset_path', type=str,
                        default='./jiuzhang_rerank_numina100k_newadj_1.jsonl',
                        help='Path to save ranked dataset')
    
    args = parser.parse_args()
    
    # Load embeddings and graph
    benchmark_content_embeds, adj_matrix, content_ls_ls = load_embeddings_and_graph(
        args.reference_embed_path, args.skill_graph_path)
    
    # Load test target data embeddings
    test_target_data_embed = np.load(args.target_embed_path)
    print(f"Test target data shape: {test_target_data_embed.shape}")
    
    # Compute similarity scores
    pooling_score = compute_similarity_scores(
        test_target_data_embed, 
        benchmark_content_embeds, 
        content_ls_ls, 
        adj_matrix
    )
    
    # Save scores
    torch.save(pooling_score, args.scores_output_path)
    
    # Load scores and sort indices
    scores = torch.load(args.scores_output_path)
    sorted_indices = torch.argsort(scores, descending=True)
    
    # Load dataset and create the sorted subset
    dataset = load_dataset('json', data_files={'train': args.target_dataset_path})
    
    sorted_set = dataset['train'].select(sorted_indices)
    sorted_set.to_json(args.ranked_dataset_path)

if __name__ == "__main__":
    main()
