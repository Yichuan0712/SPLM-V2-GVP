import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pathlib import Path
from sklearn.metrics import silhouette_score
import argparse
from matplotlib.lines import Line2D


from utils.generate_struct_embedding import generate_struct_embedding


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
    query_embedding_dic,
):
    """
    Evaluate structure embeddings on a CATH subset at multiple hierarchy levels.

    Input
    -----
    query_embedding_dic : dict / OrderedDict
        { pid(str) -> embedding(np.ndarray) }
        pid looks like "..._3.40.50" (CATH fold code after underscore)

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

    if query_embedding_dic is None or len(query_embedding_dic) == 0:
        print("No samples found in CATH embeddings; skipping evaluation.")
        return []

    # ---- build embeddings (N, D) ----
    vals = list(query_embedding_dic.values())
    try:
        seq_embeddings = np.concatenate(vals, axis=0)
    except ValueError:
        # handle case where each embedding is (D,)
        seq_embeddings = np.vstack([v.reshape(1, -1) for v in vals])

    print(f"seq_embeddings = {seq_embeddings.shape}")

    if seq_embeddings.shape[0] == 0:
        print("No samples found in CATH dataset; skipping evaluation.")
        return []

    # ---- build labels ----
    labels = []
    batch = {"pid": list(query_embedding_dic.keys())}
    labels.extend([pid.split("_")[1] for pid in batch["pid"]])

    # ---- t-SNE projection for visualization ----
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

            display_label = f"{key} ({code2name.get(key, 'Unknown')})"
            legend_labels[display_label] = color_str
            index += 1

        print(f"sample num (digit {digit_num}) = {len(select_index)}")
        if len(select_index) == 0:
            scores.extend([float("nan"), float("nan"), float("nan"), float("nan")])
            continue

        # Calinski–Harabasz on full and t-SNE embeddings
        scores.append(calinski_harabasz_score(seq_embeddings[select_index], color))  # full
        scores.append(calinski_harabasz_score(z_tsne_seq[select_index], color))      # t-SNE

        # Legend title
        if digit_num == 1:
            legend_title = "CATH class"
        elif digit_num == 2:
            legend_title = "CATH architecture"
        else:
            legend_title = "CATH fold"

        # t-SNE scatter with legend (uses your existing scatter_labeled_z)
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
        scores.append(silhouette_score(seq_embeddings[select_index], color))

    return scores


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
    parser.add_argument(
        "--truncate_inference",
        default=None,
        help="1:truncate the sequence length. 0: use full sequence length. default: None."
    )
    parser.add_argument(
        "--max_length_inference",
        default=None,
        type=int,
        nargs='?',
        help="max sequence length during inference. Default: None."
    )
    parser.add_argument(
        "--afterproject",
        action="store_true",
        help="embedding after projection layer",
    )
    parser.add_argument("--result_path", type=str, default="./")
    args = parser.parse_args()

    cathpath = args.cath_path
    out_figure_path = args.result_path
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)

    # generate embeddings by reusing generate_struct_embedding (no duplicate dataloader/model code here)
    query_embedding_dic = generate_struct_embedding(
        hdf5_path=cathpath,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        truncate_inference=args.truncate_inference,
        max_length_inference=args.max_length_inference,
        afterproject=args.afterproject,
        residue_level=False,
    )

    scores_cath = evaluate_with_cath_more_struct(
        out_figure_path=out_figure_path,
        query_embedding_dic=query_embedding_dic,
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
