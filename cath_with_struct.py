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


def scatter_labeled_z(z_batch, colors, filename="test_plot"):
    fig = plt.gcf()
    plt.switch_backend('Agg')
    fig.set_size_inches(3.5, 3.5)
    plt.clf()
    for n in range(z_batch.shape[0]):
        result = plt.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[n], s=50, marker="o", edgecolors='none')

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.savefig(filename)
    # pylab.show()


def evaluate_with_cath_more_struct(out_figure_path, device, batch_size,
                                   model, cathpath, configs
                                   # seq_mode="embedding",use_rotary_embeddings=False,
                                   ):
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)

    dataset = ProteinGraphDataset(cathpath, max_length=configs.model.esm_encoder.max_length,
                                  seq_mode=configs.model.struct_encoder.use_seq.seq_embed_mode,
                                  use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                  rotary_mode=configs.model.struct_encoder.rotary_mode,
                                  use_foldseek=configs.model.struct_encoder.use_foldseek,
                                  use_foldseek_vector=configs.model.struct_encoder.use_foldseek_vector,
                                  top_k=configs.model.struct_encoder.top_k,
                                  num_rbf=configs.model.struct_encoder.num_rbf,
                                  num_positional_embeddings=configs.model.struct_encoder.num_positional_embeddings)

    val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True, collate_fn=custom_collate)
    seq_embeddings = []
    labels = []
    # existingmodel.eval()
    for batch in val_loader:
        with torch.inference_mode():
            graph = batch["graph"].to(device)
            model_signature = signature(model.forward)
            if 'graph' in model_signature.parameters:
                _, _, features_struct, _ = model(graph=graph, mode='structure', return_embedding=True)
            else:
                _, _, features_struct, _ = model(graph, mode='structure', return_embedding=True)

            seq_embeddings.extend(features_struct.cpu().detach().numpy())
            labels.extend([id.split("_")[1] for id in batch['pid']])

    seq_embeddings = np.asarray(seq_embeddings)  # 100
    print(f"seq_embeddings  = {seq_embeddings.shape}")
    # print("shape of seq_embeddings=" + str(seq_embeddings.shape))
    mdel = TSNE(n_components=2, random_state=0, init='random', method='exact')
    # print("Projecting to 2D by TSNE\n")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)
    scores = []
    tol_class_seq = {"1": 0, "2": 1, "3": 2}
    tol_archi_seq = {"3.30": 3, "3.40": 4, "1.10": 0, "3.10": 2, "2.60": 1}
    tol_fold_seq = {"1.10.10": 0, "3.30.70": 3, "2.60.40": 2, "2.60.120": 1, "3.40.50": 4}
    for digit_num in [1, 2, 3]:  # first number of digits
        color = []
        keys = {}
        colorid = []
        colorindex = 0
        if digit_num == 1:
            ct = ["blue", "red", "black", "yellow", "orange", "green", "olive", "gray", "magenta", "hotpink", "pink",
                  "cyan", "peru", "darkgray", "slategray", "gold"]
        else:
            ct = ["black", "yellow", "orange", "green", "olive", "gray", "magenta", "hotpink", "pink", "cyan", "peru",
                  "darkgray", "slategray", "gold"]

        select_index = []
        color_dict = {}
        index = 0
        for label in labels:
            key = ".".join([x for x in label.split(".")[0:digit_num]])
            if digit_num == 1:
                keys = tol_class_seq
            if digit_num == 2:
                keys = tol_archi_seq
                if key not in tol_archi_seq:
                    index += 1
                    continue
            if digit_num == 3:
                keys = tol_fold_seq
                if key not in tol_fold_seq:
                    index += 1
                    continue

            color.append(ct[(keys[key]) % len(ct)])
            colorid.append(keys[key])
            select_index.append(index)
            color_dict[keys[key]] = ct[keys[key]]
            index += 1

        print(f"sample num={len(select_index)}")
        scores.append(calinski_harabasz_score(seq_embeddings[select_index], color))
        scores.append(calinski_harabasz_score(z_tsne_seq[select_index], color))

        scatter_labeled_z(z_tsne_seq[select_index], color,
                          filename=os.path.join(out_figure_path, f"CATHgvp_{digit_num}.png"))
        # add kmeans
        kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
        predicted_labels = kmeans.fit_predict(z_tsne_seq[select_index])
        predicted_colors = [color_dict[label] for label in predicted_labels]
        ari = adjusted_rand_score(colorid, predicted_labels)
        scores.append(ari)
        scores.append(silhouette_score(seq_embeddings[select_index], color))

    return scores  # [digit_num1_full,digit_num_2d,digit_num2_full,digit_num2_2d]


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
    args = parser.parse_args()

    # Build model & load checkpoint
    model, device, configs = StructRepresentModel(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    cathpath = configs.valid_settings.eval_struct.cath_pdb_path
    base = os.path.splitext(args.config_path)[0]
    out_figure_path = os.path.join(base, "CATH_test_release")
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)
    print("Saved to", out_figure_path)

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
