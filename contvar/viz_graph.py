import os
import random
import pickle

import torch
import networkx as nx
import wandb

from contvar.config import ProjectConfig
from contvar.data.mapper import TripletDataPathMapper, DmsProteinSplitError


def visualize_graph(protein_id=None, data_root=None, device=None):
    """Visualize a processed protein graph"""
    import matplotlib.pyplot as plt
    from graphein.protein.visualisation import plotly_protein_structure_graph

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if data_root is None:
        from contvar.config import setup_environment
        data_root = setup_environment()['data_root']

    print("Starting visualization...")

    cfg = ProjectConfig()
    try:
        mapper = TripletDataPathMapper(
            data_root,
            split_json_path=cfg.dms_protein_split_json_path,
        )
    except DmsProteinSplitError as exc:
        print(f"DMS split configuration error: {exc}")
        return

    if not mapper.triplets:
        print("No data found!")
        return

    wandb.init(
        project="ContVAR-Project",
        name="Graph-Visualization",
        job_type="visualization",
        config=vars(cfg)
    )

    if protein_id:
        choice = next((t for t in mapper.triplets if protein_id in t['anchor']), None)
        if not choice:
            print(f"Protein {protein_id} not found!")
            wandb.finish()
            return
    else:
        choice = random.choice(mapper.triplets)

    pdb_path = choice['anchor']
    pdb_code = os.path.splitext(os.path.basename(pdb_path))[0]

    processed_dir = os.path.join(data_root, "processed")
    pickle_path = os.path.join(processed_dir, f"{pdb_code}.pickle")

    print(f"Visualizing: {pdb_code}")

    if not os.path.exists(pickle_path):
        print(f"Graph not processed yet. Run training first!")
        wandb.finish()
        return

    try:
        with open(pickle_path, "rb") as f:
            g = pickle.load(f)

        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        density = nx.density(g)

        print(f"Nodes: {num_nodes}, Edges: {num_edges}, Density: {density:.4f}")

        fig = plotly_protein_structure_graph(
            g,
            colour_edges_by="kind",
            label_node_ids=False,
            node_size_multiplier=1
        )

        fig.update_layout(title=f"Graph Topology: {pdb_code}")

        wandb.log({
            "Interactive_Graph": fig,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "graph_density": density
        })

        print(f"Visualization uploaded successfully!")

    except Exception as e:
        print(f"Error: {e}")

    wandb.finish()
