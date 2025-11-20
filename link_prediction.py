import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
import numpy as np

# --- 1. Model Definition ---

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

# --- 2. Data Preparation (Synthetic Demo) ---

def create_demo_data(num_nodes=100, num_edges=300, feature_dim=32):
    """
    Creates a synthetic citation graph for demonstration.
    In a real scenario, you would load this from Neo4j or your CSVs.
    """
    print(f"Creating synthetic graph with {num_nodes} nodes and {num_edges} edges...")
    
    # Random node features (e.g., SPECTER embeddings)
    x = torch.randn(num_nodes, feature_dim)
    
    # Random edges (citations)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    
    # Split edges for Link Prediction (Train/Val/Test)
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False, # Citation graphs are directed
        add_negative_train_samples=False
    )
    train_data, val_data, test_data = transform(data)
    
    return train_data, val_data, test_data

# --- 3. Training Loop ---

def train(model, optimizer, train_data, criterion):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass (Encode)
    z = model.encode(train_data.x, train_data.edge_index)

    # Negative Sampling (for training)
    # We need negative edges (edges that don't exist) to teach the model what NOT to predict
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    # Decode (Predict scores for positive and negative edges)
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index)
    
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()

# --- 4. Evaluation ---

@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    
    # Use the edges reserved for testing (positive samples)
    # And generate negative samples
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1), method='sparse')
        
    edge_label_index = torch.cat(
        [data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    
    edge_label = torch.cat([
        data.edge_label,
        data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).sigmoid()
    
    return roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())

# --- 5. Main Execution ---

def main():
    # 1. Setup Data
    train_data, val_data, test_data = create_demo_data()
    
    # 2. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(in_channels=32, hidden_channels=64, out_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    print("\nStarting Training...")
    for epoch in range(1, 101):
        loss = train(model, optimizer, train_data, criterion)
        if epoch % 10 == 0:
            val_auc = test(model, val_data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')

    # 3. Final Evaluation
    test_auc = test(model, test_data)
    print(f'\nFinal Test AUC: {test_auc:.4f}')
    
    # 4. Predict New Links (View G Application)
    print("\n--- View G: Predicting Missing Links ---")
    z = model.encode(test_data.x, test_data.edge_index)
    
    # Let's take two random nodes and see if the model thinks they should be connected
    node_a = 0
    node_b = 5
    
    # Create a query edge
    query_edge = torch.tensor([[node_a], [node_b]], device=device)
    score = model.decode(z, query_edge).sigmoid().item()
    
    print(f"Prediction for connection Node {node_a} -> Node {node_b}: {score:.4f}")
    if score > 0.8:
        print("Recommendation: High probability of latent relation!")
    else:
        print("Recommendation: Low probability of connection.")

if __name__ == "__main__":
    main()
