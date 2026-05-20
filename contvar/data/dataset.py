import os
import glob
import random
import collections
import tempfile

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected
from Bio.PDB import MMCIFParser, PDBIO
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph

from contvar.config import ProjectConfig
from contvar.data.collate import parse_mut_pos_from_path


class TripletProteinGraphDataset(Dataset):
    """PyTorch Geometric Dataset for protein triplets with configurable edge construction.

    Supports two edge construction modes:
    - "salad": SALAD-style hybrid edges (index + spatial + random neighbors)
    - "graphein": Graphein kNN edges with Euclidean distance edge features
    """

    def __init__(self, mapper, root, config: ProjectConfig, split='train',
                 esm2_embedding_path=None, force: bool = False,
                 preloaded_embeddings=None):
        self.mapper = mapper
        self.config = config
        self.split = split
        self.triplets = mapper.get_split(split)
        self.esm2_embedding_path = esm2_embedding_path

        processed_dir = os.path.join(root, "processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        if force:
            self._clear_processed_files(root)

        # Load embeddings if needed
        self.esm2_embeddings = {}
        if preloaded_embeddings is not None:
            self.esm2_embeddings = preloaded_embeddings

        # Get active node features
        self.node_metadata_funcs = self.config.get_active_node_metadata_funcs()
        self.node_attributes = self.config.get_node_attributes_list()

        # Initialize edge builder based on mode
        if self.config.edge_mode == "salad":
            self.salad_edge_builder = self.config.get_salad_edge_builder()
            self.edge_funcs = []
            print(f"Using SALAD-style edges: index={config.salad_num_index}, "
                  f"spatial={config.salad_num_spatial}, random={config.salad_num_random}")
        else:  # graphein mode
            self.salad_edge_builder = None
            self.edge_funcs = self.config.get_active_edge_funcs()
            print(f"Using Graphein kNN edges: k={config.knn_k}")

        # Store all triplets for processing
        self.all_triplets = mapper.triplets

        super().__init__(root)

    def _clear_processed_files(self, root):
        processed_dir = os.path.join(root, "processed")
        for path in glob.glob(os.path.join(processed_dir, "*.pt")):
            os.remove(path)

    @property
    def processed_file_names(self):
        unique_paths = set()
        for t in self.all_triplets:
            unique_paths.add(t['anchor'])
            unique_paths.update(t['positives'])
            unique_paths.update(t['negatives'])
        return [os.path.basename(p).replace(".cif", ".pt") for p in unique_paths]

    @property
    def raw_file_names(self):
        return []

    def len(self) -> int:
        return len(self.triplets)

    def process(self):
        """Process all unique proteins and save to disk"""
        unique_paths = set()
        for t in self.all_triplets:
            unique_paths.add(t['anchor'])
            unique_paths.update(t['positives'])
            unique_paths.update(t['negatives'])

        print(f"Processing {len(unique_paths)} unique proteins...")

        for path in tqdm(list(unique_paths), desc="Processing"):
            pdb_code = os.path.splitext(os.path.basename(path))[0]
            pt_path = os.path.join(self.processed_dir, f"{pdb_code}.pt")

            if os.path.exists(pt_path):
                continue

            g = self._build_graph(path)
            if g is None:
                continue

            data = self._create_pyg_data(g)
            torch.save(data, pt_path)

    def _build_graph(self, path: str):
        """Build protein graph from CIF file"""
        protein_code = os.path.basename(path).replace(".cif", "")
        temp_pdb_path = None

        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(protein_code, path)

            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                temp_pdb_path = tmp.name

            pdb_io = PDBIO()
            pdb_io.set_structure(structure)
            pdb_io.save(temp_pdb_path)

            if self.config.edge_mode == "salad":
                edge_funcs = []
            else:
                edge_funcs = self.edge_funcs

            config = ProteinGraphConfig(
                edge_construction_functions=edge_funcs,
                node_metadata_functions=self.node_metadata_funcs,
                verbose=False
            )
            g = construct_graph(config=config, path=temp_pdb_path, verbose=False)

            if g is None or len(g.nodes()) == 0:
                print(f"Warning: Empty graph for {path}")
                return None

            first_node = list(g.nodes())[0]
            chain_id = first_node.split(":")[0]
            g = self._process_graph(g, chain_id, path, protein_code)
            return g

        except Exception as e:
            print(f"Failed to process {path}: {e}")
            return None
        finally:
            if temp_pdb_path and os.path.exists(temp_pdb_path):
                os.remove(temp_pdb_path)

    def _process_graph(self, g, chain_id, pdb_path, pdb_code):
        """Process graph node features and edge features."""
        protein_embedding = None
        key = pdb_code.lower()
        if key.endswith("_model"):
            key = key[:-6]
        if self.esm2_embeddings:
            if key in self.esm2_embeddings:
                protein_embedding = self.esm2_embeddings[key]
            else:
                sample_key = list(self.esm2_embeddings.keys())[0] if self.esm2_embeddings else "EMPTY"
                print(f"MISSING KEY ERROR: Could not find '{key}' in embeddings.")
                print(f"Sample key from H5: '{sample_key}'")

        # Build a sorted node order (chain, resseq) to match ESM2 embedding order
        if protein_embedding is not None:
            node_order = []
            for n, d in g.nodes(data=True):
                parts = n.split(":")
                node_chain = parts[0]
                node_resseq = int(parts[2])
                node_order.append((node_chain, node_resseq, n))
            node_order.sort(key=lambda x: (x[0], x[1]))
            name_to_emb_idx = {name: i for i, (_, _, name) in enumerate(node_order)}

        # Process nodes
        for index, (n, d) in enumerate(g.nodes(data=True)):
            aa = n.split(":")[1]
            d['chain_id'] = chain_id
            d['residue_name'] = aa
            d['residue_number'] = int(n.split(":")[2])

            if protein_embedding is not None:
                emb_idx = name_to_emb_idx[n]
                d['embedding'] = protein_embedding[emb_idx]

        # Process edges (only for graphein mode)
        if self.config.edge_mode == "graphein":
            for s, t, d in g.edges(data=True):
                source_coords = g.nodes[s]["coords"]
                target_coords = g.nodes[t]["coords"]
                d["euclidean_distance"] = round(np.sqrt(np.sum(np.square(source_coords - target_coords))).item(), 5)

        return g

    def _create_pyg_data(self, g, to_undirected_graph=True):
        """Convert NetworkX graph to PyTorch Geometric Data object"""
        if self.config.edge_mode == "salad":
            return self._create_pyg_data_salad(g, to_undirected_graph)
        else:
            return self._create_pyg_data_graphein(g, to_undirected_graph)

    def _create_pyg_data_salad(self, g, to_undirected_graph=True):
        """Create PyG Data object with SALAD-style edges"""
        node_features = collections.defaultdict(list)
        coords_list = []
        residue_indices = []
        chain_ids = []

        for index, (n, d) in enumerate(g.nodes(data=True)):
            _list = []
            for k in self.node_attributes:
                v = d.get(k)
                if v is None:
                    continue
                if isinstance(v, (list, np.ndarray)):
                    _list.extend(list(v))
                else:
                    _list.append(v)

            node_features["x"].append(_list)
            node_features["pos"].append(d["coords"].tolist())
            coords_list.append(d["coords"])
            residue_indices.append(d.get("residue_number", index))
            chain_ids.append(d.get("chain_id", "A"))

        data = Data()
        data.x = torch.tensor(node_features["x"], dtype=torch.float)
        data.pos = torch.tensor(node_features["pos"], dtype=torch.float)

        # Build edges using salad-style builder
        coords_array = np.array(coords_list)
        residue_array = np.array(residue_indices)

        unique_chains = list(set(chain_ids))
        chain_int = np.array([unique_chains.index(c) for c in chain_ids])

        edge_index, edge_attr = self.salad_edge_builder.build_edge_index_and_attr(
            coords=coords_array,
            residue_indices=residue_array,
            chain_ids=chain_int,
            use_rbf=True,
            num_rbf=self.config.salad_num_rbf,
            d_max=self.config.salad_d_max
        )

        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.residue_number = torch.tensor(residue_indices, dtype=torch.long)

        if to_undirected_graph and data.edge_index.numel() > 0:
            data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)

        return data

    def _create_pyg_data_graphein(self, g, to_undirected_graph=True):
        """Create PyG Data object with Graphein kNN edges."""
        node_indexes_mapping = {}
        node_features = collections.defaultdict(list)
        residue_numbers = []

        for index, (n, d) in enumerate(g.nodes(data=True)):
            _list = []
            for k in self.node_attributes:
                v = d.get(k)
                if v is None:
                    continue
                if isinstance(v, (list, np.ndarray)):
                    _list.extend(list(v))
                else:
                    _list.append(v)

            node_features["x"].append(_list)
            node_features["pos"].append(d["coords"].tolist())
            node_indexes_mapping[n] = index
            residue_numbers.append(d.get("residue_number", index))

        edge_features = collections.defaultdict(list)
        for s, t, d in g.edges(data=True):
            edge_features["edge_index"].append([node_indexes_mapping[s], node_indexes_mapping[t]])
            edge_features["edge_attr"].append([d["euclidean_distance"]])

        data = Data()
        data.x = torch.tensor(node_features["x"], dtype=torch.float)
        data.pos = torch.tensor(node_features["pos"], dtype=torch.float)
        data.residue_number = torch.tensor(residue_numbers, dtype=torch.long)

        if edge_features["edge_index"]:
            data.edge_index = torch.tensor(edge_features["edge_index"], dtype=torch.long).t().contiguous()
            data.edge_attr = torch.tensor(edge_features["edge_attr"], dtype=torch.float)
        else:
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
            data.edge_attr = torch.empty((0, self.config.edge_attr_dim), dtype=torch.float)

        if to_undirected_graph and data.edge_index.numel() > 0:
            data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)

        return data

    def _load_processed_graph(self, path):
        pdb_code = os.path.splitext(os.path.basename(path))[0]
        pt_path = os.path.join(self.processed_dir, f"{pdb_code}.pt")

        if os.path.exists(pt_path):
            data = torch.load(pt_path, weights_only=False)
        else:
            print(f"Warning: {pdb_code} not found, processing on the fly")
            g = self._build_graph(path)
            data = self._create_pyg_data(g) if g else None
        return data

    def get(self, idx):
        real_idx = idx % len(self.triplets)
        t = self.triplets[real_idx]

        pos_path = random.choice(t["positives"])
        data_a = self._load_processed_graph(t["anchor"])
        data_p = self._load_processed_graph(pos_path)

        if self.config.loss_type == "semi_hard":
            neg_files = t["negatives"]
            if self.config.max_negatives is not None:
                neg_files = random.sample(neg_files, min(self.config.max_negatives, len(neg_files)))
            data_n_list = [self._load_processed_graph(n) for n in neg_files]
            mut_pos_negatives = [parse_mut_pos_from_path(n) for n in neg_files]
        else:
            neg_path = random.choice(t["negatives"])
            data_n_list = [self._load_processed_graph(neg_path)]
            mut_pos_negatives = [parse_mut_pos_from_path(neg_path)]

        mut_pos_positive = parse_mut_pos_from_path(pos_path)
        return data_a, data_p, data_n_list, mut_pos_positive, mut_pos_negatives

    def download(self):
        pass


class ExhaustiveTripletDataset(Dataset):
    """
    Dataset that generates ALL positive x negative combinations for each anchor.

    This ensures the model sees every data point during warm-up phase.
    """

    def __init__(self, mapper, root, config: ProjectConfig, split='train',
                 preloaded_embeddings=None):
        self.mapper = mapper
        self.config = config
        self.split = split
        self.triplets = mapper.get_split(split)
        self.root = root

        # Build exhaustive index: list of (family_idx, pos_idx, neg_idx)
        self.exhaustive_indices = []
        for fam_idx, t in enumerate(self.triplets):
            n_pos = len(t['positives'])
            n_neg = len(t['negatives'])
            for p_idx in range(n_pos):
                for n_idx in range(n_neg):
                    self.exhaustive_indices.append((fam_idx, p_idx, n_idx))

        random.shuffle(self.exhaustive_indices)

        total_combinations = len(self.exhaustive_indices)
        n_families = len(self.triplets)
        avg_combinations = total_combinations / n_families if n_families > 0 else 0

        print(f"ExhaustiveTripletDataset ({split}):")
        print(f"  Protein families: {n_families}")
        print(f"  Total triplet combinations: {total_combinations:,}")
        print(f"  Avg combinations per family: {avg_combinations:.0f}")

        super().__init__(root)

    @property
    def processed_file_names(self):
        return []

    @property
    def raw_file_names(self):
        return []

    def len(self) -> int:
        return len(self.exhaustive_indices)

    def process(self):
        pass

    def _load_processed_graph(self, path):
        """Load processed graph from cache"""
        pdb_code = os.path.splitext(os.path.basename(path))[0]
        processed_dir = os.path.join(self.root, "processed")
        pt_path = os.path.join(processed_dir, f"{pdb_code}.pt")

        if os.path.exists(pt_path):
            data = torch.load(pt_path, weights_only=False)
        else:
            raise FileNotFoundError(f"Processed file not found: {pt_path}. Run main dataset first.")
        return data

    def get(self, idx):
        """Get a specific triplet by exhaustive index"""
        fam_idx, pos_idx, neg_idx = self.exhaustive_indices[idx]
        t = self.triplets[fam_idx]

        pos_path = t["positives"][pos_idx]
        neg_path = t["negatives"][neg_idx]
        data_a = self._load_processed_graph(t["anchor"])
        data_p = self._load_processed_graph(pos_path)
        data_n = self._load_processed_graph(neg_path)

        mut_pos_positive = parse_mut_pos_from_path(pos_path)
        mut_pos_negatives = [parse_mut_pos_from_path(neg_path)]
        return data_a, data_p, [data_n], mut_pos_positive, mut_pos_negatives

    def download(self):
        pass

    def reshuffle(self):
        """Reshuffle exhaustive indices for next epoch"""
        random.shuffle(self.exhaustive_indices)
        print(f"Reshuffled {len(self.exhaustive_indices):,} exhaustive triplets")
