import os
from collections import defaultdict, deque
import torch

class GOHierarchy:
    def __init__(self, obo_path):
        self.obo_path = obo_path
        self.parents = defaultdict(set)
        self.children = defaultdict(set)
        self.name_to_id = {}
        self.id_to_name = {}
        self.alt_ids = {}
        self._parse_obo()

    def _parse_obo(self):
        with open(self.obo_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_term = None
        is_obsolete = False
        term_id = None
        term_name = None
        term_parents = set()
        
        for line in lines:
            line = line.strip()
            if line == "[Term]":
                # Save previous term if valid
                if current_term and not is_obsolete:
                    self.id_to_name[term_id] = term_name
                    self.name_to_id[term_name] = term_id
                    for p in term_parents:
                        self.parents[term_id].add(p)
                        self.children[p].add(term_id)
                
                # Reset for new term
                current_term = True
                is_obsolete = False
                term_id = None
                term_name = None
                term_parents = set()
                continue
            
            if not current_term:
                continue

            if line.startswith("id: "):
                term_id = line[4:].strip()
            elif line.startswith("name: "):
                term_name = line[6:].strip()
            elif line.startswith("alt_id: "):
                alt_id = line[8:].strip()
                self.alt_ids[alt_id] = term_id
            elif line.startswith("is_obsolete: true"):
                is_obsolete = True
            elif line.startswith("is_a: "):
                parent_id = line[6:].split('!')[0].strip()
                term_parents.add(parent_id)
            # CAFA also sometimes includes part_of relationships for propagation, but is_a is the primary TPR constraint.
            # We will strictly use is_a for now, as is standard in CAFA.

        # Save last term
        if current_term and not is_obsolete:
            self.id_to_name[term_id] = term_name
            self.name_to_id[term_name] = term_id
            for p in term_parents:
                self.parents[term_id].add(p)
                self.children[p].add(term_id)

        # Resolve alt_ids in parents
        resolved_parents = defaultdict(set)
        for child, parent_set in self.parents.items():
            for p in parent_set:
                p_resolved = self.alt_ids.get(p, p)
                resolved_parents[child].add(p_resolved)
        self.parents = resolved_parents

    def get_ancestors(self, term_id):
        """Returns all ancestors of a term, including itself."""
        ancestors = set()
        queue = deque([term_id])
        while queue:
            curr = queue.popleft()
            if curr not in ancestors:
                ancestors.add(curr)
                queue.extend(self.parents.get(curr, []))
        return ancestors

    def build_propagation_matrix(self, vocab: dict[str, int]) -> torch.Tensor:
        """
        Builds a binary adjacency matrix A of shape (N, N) where A[i, j] = 1 
        if term_j is an ancestor of term_i (or i == j).
        
        vocab: mapping from GO term ID to column index.
        """
        N = len(vocab)
        A = torch.eye(N)
        
        vocab_keys = list(vocab.keys())
        for i, go_id in enumerate(vocab_keys):
            ancestors = self.get_ancestors(go_id)
            for anc in ancestors:
                if anc in vocab:
                    j = vocab[anc]
                    A[i, j] = 1.0
                    
        return A

    def propagate(self, preds: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Propagate probabilities upward using matrix multiplication / max pooling.
        preds: (B, N) tensor of probabilities.
        A: (N, N) binary adjacency matrix from build_propagation_matrix.
        
        For each term, its new probability is the max of its own probability and 
        the probabilities of all its descendants.
        Since A[i, j] = 1 means j is an ancestor of i, 
        A^T[j, i] = 1 means i is a descendant of j.
        """
        # We want: prop_preds[b, j] = max_{i | A[i,j]=1} preds[b, i]
        # Using broadcasting: 
        # preds is (B, N, 1)
        # A is (1, N, N)
        # masked = preds * A  => (B, N, N) where masked[b, i, j] = preds[b, i] if j is ancestor of i else 0
        # max over dim=1 (descendants i) => prop_preds[b, j]
        
        # To avoid huge memory spikes for large B and N, we can do this carefully:
        B, N = preds.shape
        device = preds.device
        A = A.to(device)
        
        prop_preds = torch.zeros_like(preds)
        for j in range(N):
            # A[:, j] is 1 for all i that are descendants of j (including j itself)
            descendants_mask = A[:, j].bool()
            if descendants_mask.any():
                prop_preds[:, j] = preds[:, descendants_mask].max(dim=1)[0]
            else:
                prop_preds[:, j] = preds[:, j]
                
        return prop_preds

if __name__ == "__main__":
    import json
    go_hierarchy = GOHierarchy('go.obo')
    print(f"Parsed {len(go_hierarchy.id_to_name)} terms from go.obo")
    
    with open('go_vocab_f.json', 'r') as f:
        vocab = json.load(f)
    
    A = go_hierarchy.build_propagation_matrix(vocab)
    print(f"Propagation matrix shape: {A.shape}")
    print(f"Number of ones in matrix: {A.sum().item()}")
