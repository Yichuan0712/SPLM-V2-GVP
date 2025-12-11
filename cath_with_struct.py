import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pathlib import Path
import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from inspect import signature
from sklearn.metrics import silhouette_score
from torch_geometric.nn import radius, global_mean_pool, global_max_pool
from data.data import custom_collate, ProteinGraphDataset
from utils.generate_struct_embedding import StructRepresentModel
import argparse
from matplotlib.lines import Line2D


def scatter_labeled_z(z_batch, colors, filename="test_plot.png",
                      legend_labels=None, legend_title=None):
    """
    Save a 2D scatter plot of embeddings colored by class.

    Parameters
    ----------
    z_batch : np.ndarray, shape [N, 2]
        2D embeddings (e.g., t-SNE output).
    colors : list[str]
        Color for each point.
    filename : str
        Output path for the PNG file.
    legend_labels : dict, optional
        Mapping from class label (str) to color (str), used to build a legend.
    legend_title : str, optional
        Title for the legend (e.g., "CATH class", "CATH architecture").
    """
    plt.switch_backend('Agg')
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    plt.clf()

    # scatter all points
    for n in range(z_batch.shape[0]):
        plt.scatter(
            z_batch[n, 0],
            z_batch[n, 1],
            c=colors[n],
            s=50,
            marker="o",
            edgecolors='none'
        )

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("z1")
    plt.ylabel("z2")

    # add legend if legend_labels is provided
    if legend_labels:
        # legend_labels: {label_str: color_str}
        handles = []
        for label_str, color_str in legend_labels.items():
            handles.append(
                Line2D(
                    [0], [0],
                    marker='o',
                    linestyle='',
                    markersize=6,
                    markerfacecolor=color_str,
                    markeredgecolor=color_str,
                    label=label_str,
                )
            )
        ax.legend(
            handles=handles,
            title=legend_title,
            loc="best",
            fontsize=6,
            title_fontsize=7,
            frameon=True,
        )

    plt.savefig(filename, bbox_inches="tight", dpi=800)


def evaluate_with_cath_more_struct(
    out_figure_path,
    device,
    batch_size,
    model,
    cathpath,
    configs,
):
    """
    Evaluate structure embeddings on a CATH subset at multiple hierarchy levels.

    This function:
      - loads preprocessed CATH HDF5 data as ProteinGraphDataset,
      - computes structure embeddings with the given model,
      - projects embeddings to 2D with t-SNE,
      - computes clustering metrics (Calinski–Harabasz, ARI, silhouette)
        at the CATH class / architecture / fold levels,
      - saves t-SNE scatter plots with color legends showing which
        color corresponds to which CATH label.

    Returns
    -------
    scores : list[float]
        [CH_full_1, CH_tsne_1, ARI_1, sil_1,
         CH_full_2, CH_tsne_2, ARI_2, sil_2,
         CH_full_3, CH_tsne_3, ARI_3, sil_3]
    """
    CATH_CLASS_NAME = {
        "1": "Mainly Alpha",
        "2": "Mainly Beta",
        "3": "Alpha Beta",
    }

    CATH_ARCHI_NAME = {
        "3.30": "2-Layer Sandwich",
        "3.40": "3-Layer(aba) Sandwich",
        "1.10": "Orthogonal Bundle",
        "3.10": "Roll",
        "2.60": "Sandwich",
    }

    CATH_FOLD_NAME = {
        "1.10.10": "Arc Repressor Mutant, subunit A",
        "3.30.70": "Alpha-Beta Plaits",
        "2.60.40": "Immunoglobulin-like",
        "2.60.120": "Jelly Rolls",
        "3.40.50": "Rossmann fold",
    }

    Path(out_figure_path).mkdir(parents=True, exist_ok=True)

    # Build dataset and dataloader
    dataset = ProteinGraphDataset(
        cathpath,
        max_length=configs.model.esm_encoder.max_length,
        seq_mode=configs.model.struct_encoder.use_seq.seq_embed_mode,
        use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
        rotary_mode=configs.model.struct_encoder.rotary_mode,
        use_foldseek=configs.model.struct_encoder.use_foldseek,
        use_foldseek_vector=configs.model.struct_encoder.use_foldseek_vector,
        top_k=configs.model.struct_encoder.top_k,
        num_rbf=configs.model.struct_encoder.num_rbf,
        num_positional_embeddings=configs.model.struct_encoder.num_positional_embeddings,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    seq_embeddings = []
    labels = []

    # Forward pass to get structure embeddings + CATH labels
    for batch in val_loader:
        with torch.inference_mode():
            graph = batch["graph"].to(device)
            model_signature = signature(model.forward)
            if "graph" in model_signature.parameters:
                _, _, features_struct, _ = model(
                    graph=graph, mode="structure", return_embedding=True
                )
            else:
                _, _, features_struct, _ = model(
                    graph, mode="structure", return_embedding=True
                )

            seq_embeddings.extend(features_struct.cpu().detach().numpy())
            # pid looks like "..._3.40.50"; we keep "3.40.50"
            labels.extend([pid.split("_")[1] for pid in batch["pid"]])

    seq_embeddings = np.asarray(seq_embeddings)
    print(f"seq_embeddings = {seq_embeddings.shape}")

    if seq_embeddings.shape[0] == 0:
        print("No samples found in CATH dataset; skipping evaluation.")
        return []

    # t-SNE projection for visualization
    tsne = TSNE(
        n_components=2,
        random_state=0,
        init="random",
        method="exact",
    )
    z_tsne_seq = tsne.fit_transform(seq_embeddings)

    scores = []

    # Hand-picked subset of CATH classes / architectures / folds
    tol_class_seq = {"1": 0, "2": 1, "3": 2}
    tol_archi_seq = {"3.30": 3, "3.40": 4, "1.10": 0, "3.10": 2, "2.60": 1}
    tol_fold_seq = {
        "1.10.10": 0,
        "3.30.70": 3,
        "2.60.40": 2,
        "2.60.120": 1,
        "3.40.50": 4,
    }

    # digit_num = 1: class; 2: architecture; 3: fold
    for digit_num in [1, 2, 3]:
        color = []
        colorid = []

        # base color palette
        if digit_num == 1:
            ct = [
                "blue", "red", "black", "yellow", "orange", "green", "olive",
                "gray", "magenta", "hotpink", "pink", "cyan", "peru",
                "darkgray", "slategray", "gold",
            ]
        else:
            ct = [
                "black", "yellow", "orange", "green", "olive", "gray",
                "magenta", "hotpink", "pink", "cyan", "peru",
                "darkgray", "slategray", "gold",
            ]

        select_index = []
        # mapping: CATH code string -> color string (for legend)
        legend_labels = {}

        index = 0

        if digit_num == 1:
            keys = tol_class_seq
            code2name = CATH_CLASS_NAME
        elif digit_num == 2:
            keys = tol_archi_seq
            code2name = CATH_ARCHI_NAME
        else:  # digit_num == 3
            keys = tol_fold_seq
            code2name = CATH_FOLD_NAME

        for label in labels:
            key = ".".join(label.split(".")[0:digit_num])
            if key not in keys:
                index += 1
                continue

            class_id = keys[key]
            color_str = ct[class_id % len(ct)]

            color.append(color_str)
            colorid.append(class_id)
            select_index.append(index)

            # display_label = code2name.get(key, key)
            # "3.40.50 (Rossmann-like)"
            display_label = f"{key} ({code2name.get(key, 'Unknown')})"

            legend_labels[display_label] = color_str
            index += 1

        print(f"sample num (digit {digit_num}) = {len(select_index)}")
        if len(select_index) == 0:
            # no samples for this level; append NaNs to keep length consistent
            scores.extend([float("nan"), float("nan"), float("nan"), float("nan")])
            continue

        # Calinski–Harabasz on full and t-SNE embeddings
        scores.append(
            calinski_harabasz_score(seq_embeddings[select_index], color)
        )  # full
        scores.append(
            calinski_harabasz_score(z_tsne_seq[select_index], color)
        )  # t-SNE

        # Legend title
        if digit_num == 1:
            legend_title = "CATH class"
        elif digit_num == 2:
            legend_title = "CATH architecture"
        else:
            legend_title = "CATH fold"

        # t-SNE scatter with legend
        scatter_labeled_z(
            z_tsne_seq[select_index],
            color,
            filename=os.path.join(out_figure_path, f"CATHgvp_{digit_num}.png"),
            legend_labels=legend_labels,
            legend_title=legend_title,
        )

        # KMeans + ARI
        kmeans = KMeans(n_clusters=len(legend_labels), random_state=42)
        predicted_labels = kmeans.fit_predict(z_tsne_seq[select_index])
        ari = adjusted_rand_score(colorid, predicted_labels)
        scores.append(ari)

        # Silhouette on full embeddings
        scores.append(
            silhouette_score(seq_embeddings[select_index], color)
        )

    return scores


def test_evaluate_allcases():
    checkpoint_path = "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/results/checkpoints_configs/checkpoint_0280000.pth"
    config_path = "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/results/checkpoints_configs/config_plddtallweight_noseq_rotary_foldseek.yaml"
    model, device, configs = StructRepresentModel(config_path=config_path,
                                                  checkpoint_path=checkpoint_path
                                                  )

    out_figure_path = os.path.join(config_path.split(".yaml")[0], "CATH_test_release")
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)
    cathpath = configs.valid_settings.eval_struct.cath_pdb_path
    scores_cath = evaluate_with_cath_more_struct(out_figure_path,
                                                 device=device,
                                                 batch_size=1,
                                                 model=model,
                                                 cathpath=cathpath,
                                                 configs=configs
                                                 # seq_mode=configs.model.struct_encoder.use_seq.seq_embed_mode,
                                                 # use_rotary_embeddings = configs.model.struct_encoder.use_rotary_embeddings
                                                 )
    print(
        f"gvp_digit_num_1:{scores_cath[0]:.4f}({scores_cath[1]:.4f})\tgvp_digit_num_2:{scores_cath[4]:.4f}({scores_cath[5]:.4f})\tgvp_digit_num_3:{scores_cath[8]:.4f}({scores_cath[9]:.4f})\n")
    print(
        f"gvp_digit_num_1_ARI:{scores_cath[2]}\tgvp_digit_num_2_ARI:{scores_cath[6]}\tgvp_digit_num_3_ARI:{scores_cath[10]}\n")
    print(
        f"gvp_digit_num_1_silhouette:{scores_cath[3]}\tgvp_digit_num_2_silhouette:{scores_cath[7]}\tgvp_digit_num_3_silhouette:{scores_cath[11]}\n")

    with open(os.path.join(out_figure_path, 'scores.txt'), 'w') as file:
        file.write(
            f"gvp_digit_num_1:{scores_cath[0]:.4f}({scores_cath[1]:.4f})\tgvp_digit_num_2:{scores_cath[3]:.4f}({scores_cath[4]:.4f})\tgvp_digit_num_3:{scores_cath[6]:.4f}({scores_cath[7]:.4f})\n")
        file.write(
            f"gvp_digit_num_1_ARI:{scores_cath[2]}\tgvp_digit_num_2_ARI:{scores_cath[5]}\tgvp_digit_num_3_ARI:{scores_cath[8]}\n")
        file.write(
            f"gvp_digit_num_1_silhouette:{scores_cath[3]}\tgvp_digit_num_2_silhouette:{scores_cath[7]}\tgvp_digit_num_3_silhouette:{scores_cath[11]}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate S-PLM2 structure embeddings on CATH."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the pretrained checkpoint (.pth).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML config used for the checkpoint.",
    )
    parser.add_argument(
        "--cath_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    # Build model & load checkpoint
    model, device, configs = StructRepresentModel(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    cathpath = args.cath_path
    base = os.path.splitext(args.config_path)[0]
    out_figure_path = os.path.join(base, "CATH_test_release")
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)

    scores_cath = evaluate_with_cath_more_struct(
        out_figure_path=out_figure_path,
        device=device,
        batch_size=1,
        model=model,
        cathpath=cathpath,
        configs=configs,
    )

    (
        ch1_full, ch1_tsne, ari1, sil1,
        ch2_full, ch2_tsne, ari2, sil2,
        ch3_full, ch3_tsne, ari3, sil3,
    ) = scores_cath

    print("\n=== CATH clustering evaluation (structure embeddings) ===")

    print(
        f"[Calinski–Harabasz score]\n"
        f"  - Class level (digit 1):       full = {ch1_full:.4f}, t-SNE = {ch1_tsne:.4f}\n"
        f"  - Architecture level (digit 2): full = {ch2_full:.4f}, t-SNE = {ch2_tsne:.4f}\n"
        f"  - Fold level (digit 3):         full = {ch3_full:.4f}, t-SNE = {ch3_tsne:.4f}"
    )

    print(
        f"\n[Adjusted Rand index (ARI) for k-means on t-SNE embeddings]\n"
        f"  - Class level (digit 1):       {ari1:.4f}\n"
        f"  - Architecture level (digit 2): {ari2:.4f}\n"
        f"  - Fold level (digit 3):         {ari3:.4f}"
    )

    print(
        f"\n[Silhouette score (using full-dimensional embeddings)]\n"
        f"  - Class level (digit 1):       {sil1:.4f}\n"
        f"  - Architecture level (digit 2): {sil2:.4f}\n"
        f"  - Fold level (digit 3):         {sil3:.4f}\n"
    )

    scores_file = os.path.join(out_figure_path, "scores.txt")
    with open(scores_file, "w") as f:
        f.write("CATH clustering evaluation (structure embeddings)\n")
        f.write("===============================================\n\n")

        f.write("[Calinski–Harabasz score]\n")
        f.write(
            f"  - Class level (digit 1):       full = {ch1_full:.4f}, t-SNE = {ch1_tsne:.4f}\n"
            f"  - Architecture level (digit 2): full = {ch2_full:.4f}, t-SNE = {ch2_tsne:.4f}\n"
            f"  - Fold level (digit 3):         full = {ch3_full:.4f}, t-SNE = {ch3_tsne:.4f}\n\n"
        )

        f.write("[Adjusted Rand index (ARI) for k-means on t-SNE embeddings]\n")
        f.write(
            f"  - Class level (digit 1):       {ari1:.4f}\n"
            f"  - Architecture level (digit 2): {ari2:.4f}\n"
            f"  - Fold level (digit 3):         {ari3:.4f}\n\n"
        )

        f.write("[Silhouette score (using full-dimensional embeddings)]\n")
        f.write(
            f"  - Class level (digit 1):       {sil1:.4f}\n"
            f"  - Architecture level (digit 2): {sil2:.4f}\n"
            f"  - Fold level (digit 3):         {sil3:.4f}\n"
        )

    print(f"Scores written to {scores_file}")


if __name__ == "__main__":
    main()
