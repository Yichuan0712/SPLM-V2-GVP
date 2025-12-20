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
from data.data import custom_collate, ProteinGraphDataset
from torch.utils.data import DataLoader

import pickle
import argparse
import os
import yaml
from typing import Dict

from accelerate import Accelerator
from collections import OrderedDict


def generate_struct_embedding_withfile(hdf5_path, query_save_path, config_path, checkpoint_path=None,
                                       truncate_inference=None,
                                       max_length_inference=None,
                                       afterproject=False,
                                       residue_level=False
                                       ):
    query_embedding_dic = generate_struct_embedding(hdf5_path=hdf5_path, config_path=config_path,
                                                    checkpoint_path=checkpoint_path,
                                                    truncate_inference=truncate_inference,
                                                    max_length_inference=max_length_inference,
                                                    afterproject=afterproject,
                                                    residue_level=residue_level)
    with open(query_save_path, 'wb') as f:
        pickle.dump(query_embedding_dic, f)

    print(f"Successfully saved embeddings to {query_save_path}")


def StructRepresentModel(config_path: str, checkpoint_path: str = None,
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
        if 'state_dict2' in checkpoint:
            state_dict = checkpoint['state_dict2']
            filtered_state_dict = {k: v for k, v in state_dict.items() if 'dummy_param' not in k}
            model.model_struct.load_state_dict(filtered_state_dict, strict=False)
        else:
            raise (f"structure's states are not in checkpoint, please check if the correct checkpoint is loaded")

    model = accelerator.prepare(model)
    model.eval()
    device = accelerator.device
    return model, device, configs


def generate_struct_embedding(hdf5_path: str, config_path: str, checkpoint_path: str = None,
                              truncate_inference=None,
                              max_length_inference=None,
                              afterproject=False,
                              residue_level=False,  # if true return residue level embeddings
                              ):
    model, device, configs = StructRepresentModel(config_path=config_path,
                                                  checkpoint_path=checkpoint_path
                                                  )
    if truncate_inference is not None:
        if truncate_inference == 0:
            max_length_inference = 100000000  # truncate_inference is 0 no truncate, set it to a very long number
    else:  # truncate_inference is None use settings from config
        if hasattr(configs.model.esm_encoder,
                   "truncate_inference") and configs.model.esm_encoder.truncate_inference:  # add this parameter to all configs True or False
            max_length_inference = configs.model.esm_encoder.max_length_inference
        else:
            max_length_inference = 100000000  # no truncate set to a very long number

    dataset = ProteinGraphDataset(hdf5_path, max_length=max_length_inference,
                                  seq_mode=configs.model.struct_encoder.use_seq.seq_embed_mode,
                                  use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                  rotary_mode=configs.model.struct_encoder.rotary_mode,
                                  use_foldseek=configs.model.struct_encoder.use_foldseek,
                                  use_foldseek_vector=configs.model.struct_encoder.use_foldseek_vector,
                                  top_k=configs.model.struct_encoder.top_k,
                                  num_rbf=configs.model.struct_encoder.num_rbf,
                                  num_positional_embeddings=configs.model.struct_encoder.num_positional_embeddings)

    val_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True, collate_fn=custom_collate)
    # query_embedding_dic = {}
    query_embedding_dic = OrderedDict()
    for batch in val_loader:
        with torch.inference_mode():
            graph = batch["graph"].to(device)
            model_signature = signature(model.forward)
            if 'graph' in model_signature.parameters:
                struct_proj, residue_proj, features_struct, features_residue = model(graph=graph, mode='structure',
                                                                                     return_embedding=True)
            else:
                struct_proj, residue_proj, features_struct, features_residue = model(graph, mode='structure',
                                                                                     return_embedding=True)

            if residue_level:
                if afterproject:
                    struct_embeddings = struct_proj.cpu().detach().numpy()
                else:
                    struct_embeddings = features_residue.cpu().detach().numpy()

            else:
                if afterproject:
                    struct_embeddings = struct_proj.cpu().detach().numpy()
                else:
                    struct_embeddings = features_struct.cpu().detach().numpy()

            pid = batch['pid'][0]
            query_embedding_dic[pid] = struct_embeddings

    return query_embedding_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch byolmodels')
    parser.add_argument("--hdf5_path", required=True,
                        help="hdf5_path"
                        )
    parser.add_argument("--result_path", required=True,
                        help="result_path"
                             "by default is None")
    parser.add_argument("--out_file", default="protein_struct_embeddings.pkl",
                        help="The name of output file, will put in --result_path folder." +
                             "default: protein_struct_embeddings.pkl"
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
    generate_struct_embedding_withfile(hdf5_path=args.hdf5_path,
                                       query_save_path=query_save_path,
                                       config_path=args.config_path,
                                       checkpoint_path=args.checkpoint_path,
                                       truncate_inference=args.truncate_inference,
                                       max_length_inference=args.max_length_inference,
                                       afterproject=args.afterproject,
                                       residue_level=args.residue_level
                                       )

"""
python3 -m utils.generate_struct_embedding --hdf5_path "/cluster/pixstor/xudong-lab/duolin/CATH/CATH_4_3_0_non-rep_gvp/" \
--config_path "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/results/checkpoints_configs/config_plddtallweight_noseq_rotary_foldseek.yaml" \
--checkpoint_path "/cluster/pixstor/xudong-lab/duolin/splm_gvpgit/results/checkpoints_configs/checkpoint_0280000.pth" \
--result_path "./" --residue_level

import pickle
with open('protein_struct_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

"""
