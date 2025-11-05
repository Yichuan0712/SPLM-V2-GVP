import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch
from pathlib import Path
from inspect import signature
from model import prepare_models
from .utils import load_configs
import pickle
import argparse
import os
import yaml
from typing import Dict

from accelerate import Accelerator
from Bio import SeqIO


def read_fasta(fn_fasta):
    prot2seq = {}
    with open(fn_fasta) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            prot = record.id
            prot2seq[prot] = seq
    return list(prot2seq.keys()), prot2seq


def generate_seq_embedding_withfile(input_seq, query_save_path, config_path, checkpoint_path=None,
                                    truncate_inference=None,
                                    max_length_inference=None,
                                    afterproject=False,
                                    residue_level=False,
                                    ):
    _, query_sequence = read_fasta(input_seq)
    query_embedding_dic = generate_seq_embedding(query_sequence=query_sequence, config_path=config_path,
                                                 checkpoint_path=checkpoint_path,
                                                 truncate_inference=truncate_inference,
                                                 max_length_inference=max_length_inference, afterproject=afterproject,
                                                 residue_level=residue_level)
    with open(query_save_path, 'wb') as f:
        pickle.dump(query_embedding_dic, f)

    print(f"Successfully saved embeddings to {query_save_path}")


def SequenceRepresentModel(config_path: str, checkpoint_path: str = None,
                           truncate_inference=None,
                           max_length_inference=None,
                           ):
    with open(config_path) as file:
        config_file = yaml.full_load(file)
        configs = load_configs(config_file, args=None)

    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
    )

    model = prepare_models(configs=configs)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.model_seq.load_state_dict(checkpoint['state_dict1'], strict=False)

    model = accelerator.prepare(model)
    model.eval()

    alphabet = model.model_seq.alphabet
    if truncate_inference is not None:
        if truncate_inference:  # truncation_seq_length
            assert max_length_inference is not None, "truncate_inference is set to True, max_length_inference cannot be None"
            batch_converter = alphabet.get_batch_converter(
                truncation_seq_length=max_length_inference)  # add this parameter to all configs
            print(f"truncate_inference with max_length={max_length_inference}")
        else:
            batch_converter = alphabet.get_batch_converter()
    else:
        if hasattr(configs.model.esm_encoder,
                   "truncate_inference") and configs.model.esm_encoder.truncate_inference:  # add this parameter to all configs True or False
            batch_converter = alphabet.get_batch_converter(
                truncation_seq_length=configs.model.esm_encoder.max_length_inference)  # add this parameter to all configs
            print(f"truncate_inference with max_length={max_length_inference}")
        else:
            batch_converter = alphabet.get_batch_converter()  # no truncate and no max_length
    device = accelerator.device
    return model, batch_converter, device


def generate_seq_embedding(query_sequence: Dict[str, str], config_path: str, checkpoint_path: str = None,
                           truncate_inference=None,
                           max_length_inference=None,
                           afterproject=False,
                           residue_level=False,
                           ):
    model, batch_converter, device = SequenceRepresentModel(config_path=config_path,
                                                            checkpoint_path=checkpoint_path,
                                                            truncate_inference=truncate_inference,
                                                            max_length_inference=max_length_inference,
                                                            )

    query_embedding_dic = {}
    for id, seq in tqdm(zip(query_sequence.keys(), query_sequence.values()), total=len(query_sequence)):
        batch_seq = [(id, seq)]
        with torch.inference_mode():
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
            batch_tokens = batch_tokens.to(device)
            model_signature = signature(model.forward)
            if 'batch_tokens' in model_signature.parameters:
                seq_proj, residue_proj, features_seq, features_residue = model(batch_tokens=batch_tokens,
                                                                               mode='sequence', return_embedding=True)
            else:
                seq_proj, residue_proj, features_seq, features_residue = model(batch_tokens, mode='sequence',
                                                                               return_embedding=True)

            if residue_level:
                if afterproject:
                    query_embedding_dic[id] = residue_proj.cpu().detach().numpy()
                else:
                    query_embedding_dic[id] = features_residue.cpu().detach().numpy()
            else:
                if afterproject:
                    query_embedding_dic[id] = seq_proj.cpu().detach().numpy()
                else:
                    query_embedding_dic[id] = features_seq.cpu().detach().numpy()

    return query_embedding_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch byolmodels')
    parser.add_argument("--input_seq", required=True,
                        help="input_seq"
                        )
    parser.add_argument("--result_path", required=True,
                        help="result_path"
                             "by default is None")
    parser.add_argument("--out_file", default="protein_embeddings.pkl",
                        help="The name of output file, will put in --result_path folder." +
                             "default: protein_embeddings.pkl"
                        )
    parser.add_argument("--config_path", "-c", help="The location of config file", required=True)
    parser.add_argument("--num_end_adapter_layers", default=None, help="num_end_adapter_layers")
    parser.add_argument("--checkpoint_path", default=None,
                        help="path to checkpint, if not provode use ESM2's checkpoint, indicates in the config_file")

    parser.add_argument("--truncate_inference", default=None,
                        help="1:truncate the sequence length. 0: use full sequence length. default: None." +
                             "if None use setting in config file, if not None, overwrite the setting in config file")
    parser.add_argument('--afterproject', action='store_true', help='embedding after projection layer')
    parser.add_argument(
        "--max_length_inference",
        default=None,
        type=int,
        nargs='?',  # Make the argument optional
        help="max sequence length during inference. Default: None. If None, use setting in config file. If not None, overwrite the setting in config file."
    )
    parser.add_argument('--residue_level', action='store_true', help='return residue_level embedding')
    args = parser.parse_args()
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    query_save_path = os.path.join(args.result_path, args.out_file)
    if args.residue_level:
        print("Generating residue level representations")

    generate_seq_embedding_withfile(input_seq=args.input_seq,
                                    query_save_path=query_save_path,
                                    config_path=args.config_path,
                                    checkpoint_path=args.checkpoint_path,
                                    truncate_inference=args.truncate_inference,
                                    max_length_inference=args.max_length_inference,
                                    afterproject=args.afterproject,
                                    residue_level=args.residue_level,
                                    )

"""
python3 -m utils.generate_seq_embedding --input_seq "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/PLMsearch/plmsearch_data/swissprot_to_swissprot_5/protein.fasta" \
--config_path "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/results/checkpoints_configs/config_plddtallweight_noseq_rotary_foldseek.yaml" \
--checkpoint_path "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/results/checkpoints_configs/checkpoint_0280000.pth" \
--result_path "./"

#truncate_inference with max_length_inference=1022
python3 -m utils.generate_seq_embedding --input_seq "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/PLMsearch/plmsearch_data/swissprot_to_swissprot_5/protein.fasta" \
--config_path "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/results/checkpoints_configs/config_plddtallweight_noseq_rotary_foldseek.yaml" \
--checkpoint_path "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/results/checkpoints_configs/checkpoint_0280000.pth" \
--result_path "./" --out_file "truncate_protein_embeddings.pkl" \
--truncate_inference 1 --max_length_inference 1022

#residue_level representations
python3 -m utils.generate_seq_embedding --input_seq "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/PLMsearch/plmsearch_data/swissprot_to_swissprot_5/protein.fasta" \
--config_path "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/results/checkpoints_configs/config_plddtallweight_noseq_rotary_foldseek.yaml" \
--checkpoint_path "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/results/checkpoints_configs/checkpoint_0280000.pth" \
--result_path "./" --out_file "truncate_protein_embeddings.pkl" \
--truncate_inference 1 --max_length_inference 1022 --residue_level

import pickle
with open('truncate_protein_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

"""
