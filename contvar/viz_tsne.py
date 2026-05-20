import os
import random

import numpy as np
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from contvar.data.collate import parse_mut_pos_from_path


def _load_graph(path, processed_dir):
    """Load a processed .pt graph by CIF path."""
    pdb_code = os.path.splitext(os.path.basename(path))[0]
    pt_path = os.path.join(processed_dir, f"{pdb_code}.pt")
    if os.path.exists(pt_path):
        return torch.load(pt_path, weights_only=False)
    return None


def _collect_embeddings(model, mapper, processed_dir, device,
                        split='val', max_samples=3000, n_proteins=10, seed=42):
    """Collect raw and projected embeddings from processed graphs for t-SNE."""
    triplets = mapper.get_split(split)

    # Select top N protein families by variant count
    family_sizes = []
    for i, t in enumerate(triplets):
        total = len(t['positives']) + len(t['negatives'])
        if total > 0:
            family_sizes.append((i, total, t['protein_id']))
    family_sizes.sort(key=lambda x: -x[1])
    selected = family_sizes[:n_proteins]

    model.eval()

    # Mutant data
    raw_global, raw_local = [], []
    proj_global, proj_local = [], []
    labels, protein_ids = [], []

    # Wild-type data
    wt = {'raw_g': [], 'raw_l': [], 'proj_g': [], 'proj_l': [], 'ids': []}

    with torch.no_grad():
        for fam_idx, _, prot_id in selected:
            t = triplets[fam_idx]

            # --- Anchor (wild-type) ---
            anchor_data = _load_graph(t['anchor'], processed_dir)
            if anchor_data is None:
                continue

            batch_a = Batch.from_data_list([anchor_data]).to(device)
            raw_g = global_mean_pool(batch_a.x.float(), batch_a.batch)
            wt['raw_g'].append(raw_g.cpu().numpy()[0])
            wt['raw_l'].append(raw_g.cpu().numpy()[0])  # no mutation pos for WT

            eg, el = model(batch_a)
            wt['proj_g'].append(eg.cpu().numpy()[0])
            wt['proj_l'].append(el.cpu().numpy()[0])
            wt['ids'].append(prot_id)

            # --- Variants (positives=benign, negatives=pathogenic) ---
            variants = [(p, 1) for p in t['positives']] + [(n, 0) for n in t['negatives']]
            for path, label in variants:
                data = _load_graph(path, processed_dir)
                if data is None:
                    continue

                mut_pos = parse_mut_pos_from_path(path)
                mut_pos_t = torch.tensor(
                    [mut_pos if mut_pos is not None else -1], dtype=torch.long
                )

                batch = Batch.from_data_list([data]).to(device)

                # Raw baseline: mean-pooled node features (global) and mutation-site feature (local)
                x_float = batch.x.float()
                rg = global_mean_pool(x_float, batch.batch)
                raw_global.append(rg.cpu().numpy()[0])

                if (mut_pos is not None
                        and hasattr(batch, 'residue_number')
                        and batch.residue_number is not None):
                    mask = batch.residue_number == mut_pos
                    if mask.any():
                        rl = x_float[mask][0]
                    else:
                        rl = rg[0]
                else:
                    rl = rg[0]
                raw_local.append(rl.cpu().numpy().flatten())

                # Projected embeddings through the GNN
                eg, el = model(batch, mut_pos=mut_pos_t.to(device))
                proj_global.append(eg.cpu().numpy()[0])
                proj_local.append(el.cpu().numpy()[0])

                labels.append(label)
                protein_ids.append(prot_id)

    raw_global = np.array(raw_global)
    raw_local = np.array(raw_local)
    proj_global = np.array(proj_global)
    proj_local = np.array(proj_local)
    labels = np.array(labels)
    protein_ids = np.array(protein_ids)

    for k in ['raw_g', 'raw_l', 'proj_g', 'proj_l']:
        wt[k] = np.array(wt[k]) if wt[k] else np.empty((0, raw_global.shape[1] if k.endswith('g') else raw_local.shape[1]))

    # Subsample if too large
    if len(raw_global) > max_samples:
        np.random.seed(seed)
        idx = np.random.choice(len(raw_global), max_samples, replace=False)
        raw_global, raw_local = raw_global[idx], raw_local[idx]
        proj_global, proj_local = proj_global[idx], proj_local[idx]
        labels, protein_ids = labels[idx], protein_ids[idx]
        print(f"[t-SNE] Subsampled to {max_samples} points")

    return {
        'raw_global': raw_global, 'raw_local': raw_local,
        'proj_global': proj_global, 'proj_local': proj_local,
        'labels': labels, 'protein_ids': protein_ids, 'wt': wt,
    }


def visualize_tsne(model, mapper, processed_dir, device,
                   save_dir="visualizations", split='val',
                   max_samples=3000, perplexity=30, seed=42, n_proteins=10):
    """Generate t-SNE plots comparing baseline vs projected embeddings.

    Produces two figures:
      1. 2x2 comparison: Baseline vs Projected x Global vs Local
         (mutants colored red/green by pathogenicity, WT as gold markers)
      2. Per-protein clustering: Baseline vs Projected (global), colored by protein family

    Args:
        model: Trained DeepProteinGAT model.
        mapper: TripletDataPathMapper with split information.
        processed_dir: Path to directory with processed .pt graph files.
        device: torch device.
        save_dir: Directory to save plots.
        split: Which data split to visualize ('train' or 'val').
        max_samples: Maximum number of mutant samples for t-SNE.
        perplexity: t-SNE perplexity parameter.
        seed: Random seed for reproducibility.
        n_proteins: Number of top protein families to include.

    Returns:
        Tuple of (comparison_plot_path, per_protein_plot_path).
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[t-SNE] Collecting embeddings for top {n_proteins} proteins ({split} split)...")
    emb = _collect_embeddings(
        model, mapper, processed_dir, device,
        split=split, max_samples=max_samples, n_proteins=n_proteins, seed=seed
    )

    labels = emb['labels']
    protein_ids = emb['protein_ids']
    wt = emb['wt']
    unique_proteins = list(dict.fromkeys(protein_ids))  # preserve order

    print(f"[t-SNE] {len(labels)} mutants, {len(wt['ids'])} wild-types, "
          f"{len(unique_proteins)} protein families")

    # --- Marker assignment ---
    filled_markers = ["o", "s", "D", "v", "^", "<", "p", "P", "*", "X", "h", "H", "8", "d"]
    unfilled_markers = ["1", "2", "3", "4", "+", "|", "_"]
    all_markers = filled_markers + unfilled_markers
    random.seed(seed)
    random.shuffle(all_markers)
    marker_map = {pid: all_markers[i % len(all_markers)] for i, pid in enumerate(unique_proteins)}

    # ============================
    # 1. Comparison Plot (2x2)
    # ============================
    n_wt = len(wt['ids'])
    datasets = {
        'Baseline Global (Raw Features)': {
            'X': np.vstack([emb['raw_global'], wt['raw_g']]) if n_wt else emb['raw_global'],
            'n_wt': n_wt,
        },
        'Projected Global': {
            'X': np.vstack([emb['proj_global'], wt['proj_g']]) if n_wt else emb['proj_global'],
            'n_wt': n_wt,
        },
        'Baseline Local (Raw Features)': {
            'X': emb['raw_local'],
            'n_wt': 0,
        },
        'Projected Local': {
            'X': emb['proj_local'],
            'n_wt': 0,
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle(
        f't-SNE Embedding Visualization (Top {n_proteins} Proteins)\n'
        f'Baseline vs Projected \u00b7 Global vs Local',
        fontsize=18, fontweight='bold', y=0.98
    )

    for ax, (title, dset) in zip(axes.flatten(), datasets.items()):
        print(f"[t-SNE] Computing: {title}...")
        n_pts = len(dset['X'])
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, n_pts - 1),
            random_state=seed, max_iter=1000, init='pca', learning_rate='auto'
        )
        emb_2d = tsne.fit_transform(dset['X'])

        n_wt = dset['n_wt']
        n_mut = n_pts - n_wt
        mut_2d, wt_2d = emb_2d[:n_mut], emb_2d[n_mut:]

        # Plot mutants per protein (red=pathogenic, green=benign)
        for pid in unique_proteins:
            marker = marker_map[pid]
            is_filled = marker not in unfilled_markers
            ec = 'none' if is_filled else None

            mask_p = protein_ids == pid
            for label_val, color in [(0, '#e74c3c'), (1, '#2ecc71')]:
                mask = (labels == label_val) & mask_p
                if np.any(mask):
                    ax.scatter(
                        mut_2d[mask, 0], mut_2d[mask, 1],
                        c=color, marker=marker, s=40, alpha=0.4,
                        edgecolors=ec, linewidth=1.2 if not is_filled else 0.5
                    )

        # Plot WT as gold markers
        for i, pid in enumerate(wt['ids']):
            if i >= len(wt_2d):
                break
            marker = marker_map.get(pid, 'o')
            is_filled = marker not in unfilled_markers
            ax.scatter(
                wt_2d[i, 0], wt_2d[i, 1],
                c='gold', s=180, marker=marker,
                edgecolors='black' if is_filled else None,
                linewidth=1.2, zorder=10
            )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = os.path.join(save_dir, f"tsne_comparison_{split}_top_{n_proteins}.png")
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    plt.close()

    # ============================
    # 2. Per-protein Clustering Plot
    # ============================
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 9))
    fig2.suptitle(
        f't-SNE \u2013 Per-Protein Clustering (Top {n_proteins} Proteins)',
        fontsize=16, fontweight='bold'
    )

    cmap = plt.cm.get_cmap('tab10' if n_proteins <= 10 else 'tab20', n_proteins)

    for ax, (title, X_emb) in zip(axes2, [
        ('Baseline Global', emb['raw_global']),
        ('Projected Global', emb['proj_global']),
    ]):
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(X_emb) - 1),
            random_state=seed, max_iter=1000, init='pca', learning_rate='auto'
        )
        emb_2d = tsne.fit_transform(X_emb)

        for ci, pid in enumerate(unique_proteins):
            marker = marker_map[pid]
            is_filled = marker not in unfilled_markers
            mask_prot = protein_ids == pid

            ax.scatter(
                emb_2d[mask_prot, 0], emb_2d[mask_prot, 1],
                c=[cmap(ci)], marker=marker, s=35, alpha=0.7,
                label=pid,
                edgecolors='none' if is_filled else None
            )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8, ncol=2)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path2 = os.path.join(save_dir, f"tsne_per_protein_{split}_top_{n_proteins}.png")
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Saved Comparison: {path1}")
    print(f"[SUCCESS] Saved Per-Protein: {path2}")
    return path1, path2
