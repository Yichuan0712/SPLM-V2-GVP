import torch.nn as nn
import torch
import torch.nn.functional as F
import esm
import esm_adapterH
from peft import PeftModel, LoraConfig, get_peft_model
import gvp.models
from torch_geometric.nn import radius, global_mean_pool, global_max_pool


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        pe = pe.unsqueeze(0).transpose(0, 1)  # Reshape for batch and sequence length
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pe[:x.size(0), :]
        return x


def patch_num_32(n):
    remainder = n % 32
    if remainder == 0:
        return n // 32
    else:
        return (n + (32 - remainder)) // 32


def save_lora_checkpoint(model, save_path):
    """
    Save lora weights as a checkpoint in the save_path directory.
    """
    model.save_pretrained(save_path)


def load_and_add_lora_checkpoint(base_model, lora_checkpoint_path):
    """Add a pretrained LoRa checkpoint to a base model"""
    lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
    return lora_model


def merge_lora_with_base_model(lora_model):
    """
    This method merges the LoRa layers into the base model.
    This is needed if someone wants to use the base model as a standalone model.
    """
    model = lora_model.merge_and_unload()
    return model


def remove_lora(lora_model):
    """
    Gets back the base model by removing all the lora modules without merging
    """
    base_model = lora_model.unload()
    return base_model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_negative_mean_logtis(logits, mode, minibatch_size):
    N = minibatch_size
    if mode == "struct_struct":
        a11 = logits[0:N, 1:N]
        return a11.mean().item()

    if mode == "struct_seq":
        a12 = logits[0:N, N:2 * N - 1]
        return a12.mean().item()

    if mode == "seq_struct":
        a21 = logits[N:2 * N, 1:N]
        return a21.mean().item()
    if mode == "seq_seq":
        a22 = logits[N:2 * N, N:2 * N - 1]
        return a22.mean().item()


class MoBYMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYMLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim,
                                    out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        # print("mlp forward")
        # print(x.shape)  #[128,512,100]
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        return x


class GVPEncoder(nn.Module):  # embedding table can be tuned
    def __init__(self, configs=None, residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2,
                 seqlen=512):
        super(GVPEncoder, self).__init__()
        if hasattr(configs.model.struct_encoder, "version") and configs.model.struct_encoder.version == 'v2':
            node_in_dim = [12 + 11, 3]  ##use data_v2.py to extract gvp features
        else:
            node_in_dim = [6, 3]  # default

        if configs.model.struct_encoder.use_foldseek:
            node_in_dim[0] += 10  # foldseek has 10 more node scalar features

        if configs.model.struct_encoder.use_foldseek_vector:
            node_in_dim[1] += 6  # foldseek_vector has 6 more node vector features

        node_in_dim = tuple(node_in_dim)
        if configs.model.struct_encoder.use_rotary_embeddings:
            if configs.model.struct_encoder.rotary_mode == 3:
                edge_in_dim = (
                configs.model.struct_encoder.num_rbf + 8, 1)  # 16+2+3+3 only for mode ==3 add 8D pos_embeddings
            else:
                edge_in_dim = (configs.model.struct_encoder.num_rbf + 2, 1)  # 16+2
        else:
            edge_in_dim = (
            configs.model.struct_encoder.num_rbf + configs.model.struct_encoder.num_positional_embeddings,
            1)  # num_rbf+num_positional_embeddings

        if hasattr(configs.model.struct_encoder, "version") and configs.model.struct_encoder.version == 'v2':
            edge_in_dim = (edge_in_dim[0] + 7, 1)  # in v2 add 7 more D

        node_h_dim = configs.model.struct_encoder.node_h_dim
        # node_h_dim=(100, 16) #default
        # node_h_dim = (100, 32)  # seems best?
        edge_h_dim = configs.model.struct_encoder.edge_h_dim
        # edge_h_dim = (32, 1) #default
        gvp_num_layers = configs.model.struct_encoder.gvp_num_layers
        # gvp_num_layers = 3

        self.use_seq = configs.model.struct_encoder.use_seq.enable
        if self.use_seq:
            self.seq_embed_mode = configs.model.struct_encoder.use_seq.seq_embed_mode
            self.backbone = gvp.models.structure_encoder(node_in_dim, node_h_dim,
                                                         edge_in_dim, edge_h_dim, seq_in=True,
                                                         seq_embed_mode=self.seq_embed_mode,
                                                         seq_embed_dim=configs.model.struct_encoder.use_seq.seq_embed_dim,
                                                         num_layers=gvp_num_layers)
        else:
            self.backbone = gvp.models.structure_encoder(node_in_dim, node_h_dim,
                                                         edge_in_dim, edge_h_dim, seq_in=False,
                                                         num_layers=gvp_num_layers)

        # pretrained_state_dict = init_model.state_dict()
        # self.backbone.load_state_dict(pretrained_state_dict,strict=False)

        dim_mlp = node_h_dim[0]
        # self.esm_pool_mode= configs.model.esm_encoder.pool_mode
        self.projectors_protein = MoBYMLP(in_dim=dim_mlp, inner_dim=protein_inner_dim, out_dim=protein_out_dim,
                                          num_layers=protein_num_projector)

        self.projectors_residue = MoBYMLP(in_dim=dim_mlp, inner_dim=residue_inner_dim, out_dim=residue_out_dim,
                                          num_layers=residue_num_projector)

        if hasattr(configs.model.struct_encoder, "fine_tuning") and not configs.model.struct_encoder.fine_tuning.enable:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

        if hasattr(configs.model.struct_encoder,
                   "fine_tuning_projct") and not configs.model.struct_encoder.fine_tuning_projct.enable:
            for name, param in self.projectors_protein.named_parameters():
                param.requires_grad = False

            for name, param in self.projectors_residue.named_parameters():
                param.requires_grad = False

        # for p in self.backbone.parameters():
        #    p.requires_grad = False  # fix all previous layers

        # for name, module in model.named_modules():
        #     print(f"Module Name: {name}")
        #     print(module)
        #     print("=" * 50)

    def forward(self, graph, esm2_representation=None,
                return_embedding=False):  # this batch is torch_geometric batch.batch to indicate the batch
        """
        graph: torch_geometric batchdasta
        """
        # print("graph.node shapes")
        # print(graph.node_s.shape) #torch.Size([20650, 6])         torch.Size([16865, 23])
        # print(graph.node_v.shape) #torch.Size([20650, 9, 3])      torch.Size([16865, 9, 3])
        # print(graph.edge_s.shape) #torch.Size([619500, 18])     torch.Size([443846, 25])
        # print(graph.edge_v.shape) #torch.Size([619500, 1, 3])    torch.Size([443846, 1, 3])

        nodes = (graph.node_s, graph.node_v)
        edges = (graph.edge_s, graph.edge_v)
        if self.use_seq and self.seq_embed_mode != "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges,
                                                      seq=graph.seq)
        elif self.use_seq and self.seq_embed_mode == "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges,
                                                      seq=esm2_representation  # `torch.Tensor` of shape [num_nodes]
                                                      )
        else:
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges,
                                                      seq=None)

        # graph_feature=scatter_mean(residue_feature, batch, dim=0)
        graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph.batch)
        with torch.cuda.amp.autocast():  # Automatically handles FP16 and FP32 operations
            graph_feature = self.projectors_protein(graph_feature_embedding)
            residue_feature = self.projectors_residue(residue_feature_embedding)
        if return_embedding:
            return graph_feature, residue_feature, graph_feature_embedding, residue_feature_embedding
        else:
            return graph_feature, residue_feature


class ESM2(nn.Module):  # embedding table is fixed
    def __init__(self, esm2_pretrain,
                 configs,
                 logging=None,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(ESM2, self).__init__()
        if configs.model.esm_encoder.adapter_h.enable:
            if logging is not None:
                logging.info("use adapter H")
            # num_end_adapter_layers = configs.model.esm_encoder.adapter_h.num_end_adapter_layers
            adapter_args = configs.model.esm_encoder.adapter_h
            esm2_dict = {
                "esm2_t36_3B_UR50D": esm_adapterH.pretrained.esm2_t36_3B_UR50D(adapter_args),
                # 36 layers embedding=2560
                "esm2_t33_650M_UR50D": esm_adapterH.pretrained.esm2_t33_650M_UR50D(adapter_args),
                # 33 layers embedding=1280
                "esm2_t30_150M_UR50D": esm_adapterH.pretrained.esm2_t30_150M_UR50D(adapter_args),
                # 30 layers embedding=640
                "esm2_t12_35M_UR50D": esm_adapterH.pretrained.esm2_t12_35M_UR50D(adapter_args),
                # 12 layers embedding=480
                "esm2_t6_8M_UR50D": esm_adapterH.pretrained.esm2_t6_8M_UR50D(adapter_args),
                # 6 layers embedding = 320
            }
        else:
            esm2_dict = {
                "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D(),  # 36 layers embedding=2560
                "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D(),  # 33 layers embedding=1280
                "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D(),  # 30 layers embedding=640
                "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D(),  # 12 layers embedding=480
                "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D(),  # 6 layers embedding = 320
            }

        # if esm2_pretrain_local is None:
        self.esm2, self.alphabet = esm2_dict[esm2_pretrain]  # alphabet.all_toks
        # else:
        #     print("load esm2 model from local dir")
        #     self.esm2, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_pretrain_local)

        self.num_layers = self.esm2.num_layers
        for p in self.esm2.parameters():  # frozen all parameters first
            p.requires_grad = False

        if configs.model.esm_encoder.adapter_h.enable:
            for name, param in self.esm2.named_parameters():
                if "adapter_layer" in name:
                    # print("unfix adapter_layer")
                    param.requires_grad = True

        if configs.model.esm_encoder.lora.enable:
            if logging is not None:
                logging.info('use lora for esm v2')
            if configs.model.esm_encoder.lora.resume.enable:
                self.esm2 = load_and_add_lora_checkpoint(self.esm2, configs.model.esm_encoder.lora.resume)
            else:
                # target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]
                lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]
                target_modules = []
                if configs.model.esm_encoder.lora.esm_num_end_lora > 0:
                    start_layer_idx = np.max([self.num_layers - configs.model.esm_encoder.lora.esm_num_end_lora, 0])
                    for idx in range(start_layer_idx, self.num_layers):
                        for layer_name in lora_targets:
                            target_modules.append(f"layers.{idx}.{layer_name}")

                peft_config = LoraConfig(
                    inference_mode=False,
                    r=configs.model.esm_encoder.lora.r,
                    lora_alpha=configs.model.esm_encoder.lora.alpha,
                    target_modules=target_modules,
                    lora_dropout=configs.model.esm_encoder.lora.dropout,
                    bias="none",
                    # modules_to_save=modules_to_save
                )
                self.peft_model = get_peft_model(self.esm2, peft_config)
        elif configs.model.esm_encoder.fine_tuning.enable:
            if logging is not None:
                logging.info('fine-tune esm v2')
            unfix_last_layer = configs.model.esm_encoder.fine_tuning.unfix_last_layer
            fix_layer_num = self.num_layers - unfix_last_layer
            fix_layer_index = 0
            for layer in self.esm2.layers:  # only fine-tune transformer layers,no contact_head and other parameters
                if fix_layer_index < fix_layer_num:
                    fix_layer_index += 1  # keep these layers frozen
                    continue

                for p in layer.parameters():
                    p.requires_grad = True

            if unfix_last_layer != 0:  # if need fine-tune last layer, the emb_layer_norm_after for last representation should updated
                for p in self.esm2.emb_layer_norm_after.parameters():
                    p.requires_grad = True

        if configs.model.esm_encoder.tune_ESM_table:
            if logging is not None:
                logging.info("fine-tune esm embedding parameters")
            for p in self.esm2.embed_tokens.parameters():
                p.requires_grad = True

        if hasattr(configs.model.esm_encoder, "MLM"):
            if configs.model.esm_encoder.MLM.enable and configs.model.esm_encoder.MLM.mode == "predict":
                for p in self.esm2.lm_head.parameters():
                    p.requires_grad = True

        self.pool_mode = configs.model.esm_encoder.pool_mode
        self.projectors_residue = MoBYMLP(in_dim=self.esm2.embed_dim,
                                          inner_dim=residue_inner_dim,
                                          num_layers=residue_num_projector,
                                          out_dim=residue_out_dim)

        self.projectors_protein = MoBYMLP(in_dim=self.esm2.embed_dim,
                                          inner_dim=protein_inner_dim,
                                          num_layers=protein_num_projector,
                                          out_dim=protein_out_dim)

    def forward(self, x, return_logits=False, return_embedding=False):
        # print(x)
        # print(x.shape)
        outputs = self.esm2(x, repr_layers=[self.num_layers], return_contacts=False)
        # print("OKOKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
        if return_logits:
            prediction_scores = outputs["logits"]
            return prediction_scores
        else:
            residue_feature = outputs['representations'][self.num_layers]
            # residue_dim = residue_feature.shape #[batch,L,D]
            # average pooling but remove padding tokens
            mask = ((x != self.alphabet.padding_idx) & (x != self.alphabet.cls_idx) & (
                    x != self.alphabet.eos_idx))
            denom = torch.sum(mask, -1, keepdim=True)
            graph_feature_embedding = torch.sum(residue_feature * mask.unsqueeze(-1), dim=1) / denom  # remove padding
            graph_feature = self.projectors_protein(graph_feature_embedding)  # the cls and eos token is included
            mask = ((x != self.alphabet.padding_idx) & (x != self.alphabet.cls_idx) & (
                    x != self.alphabet.eos_idx))  # use this in v2 training
            residue_feature_embedding = residue_feature[mask]
            # print(residue_feature.shape)
            # print(residue_feature_embedding.shape)
            residue_feature = self.projectors_residue(residue_feature_embedding)
            # residue_feature = self.projectors_residue(residue_feature.view(-1, residue_dim[-1])).view(residue_dim[0],residue_dim[1],-1)
            if return_embedding:
                return graph_feature, residue_feature, graph_feature_embedding, residue_feature_embedding
            else:
                return graph_feature, residue_feature


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = (F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) + 1) / 2
        return adj


def reset_current_samelen_batch():
    batch_seq_samelen = []
    images_samelen = []
    return batch_seq_samelen, images_samelen


def check_samelen_batch(batch_size, batch_seq_samelen):
    if len(batch_seq_samelen) == batch_size:
        return True
    else:
        return False


def reset_current_mixlen_batch():
    batch_seq_mixlen = []
    images_mixlen = []
    return batch_seq_mixlen, images_mixlen


def check_mixlen_batch(batch_size, batch_seq_mixlen):
    if len(batch_seq_mixlen) == batch_size:
        return True
    else:
        return False


class MaskedLMDataCollator:
    """Data collator for masked language modeling.
    
    The idea is based on the implementation of DataCollatorForLanguageModeling at
    https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L751C7-L782
    """

    def __init__(self, batch_converter, mlm_probability=0.15):
        """_summary_

        Args:
            mlm_probability (float, optional): The probability with which to (randomly) mask tokens in the input. Defaults to 0.15.
        """
        self.mlm_probability = mlm_probability
        self.special_token_indices = [batch_converter.alphabet.cls_idx,
                                      batch_converter.alphabet.padding_idx,
                                      batch_converter.alphabet.eos_idx,
                                      batch_converter.alphabet.unk_idx,
                                      batch_converter.alphabet.mask_idx]
        self.vocab_size = batch_converter.alphabet.all_toks.__len__()
        self.mask_idx = batch_converter.alphabet.mask_idx

    def get_special_tokens_mask(self, tokens):
        return [1 if token in self.special_token_indices else 0 for token in tokens]

    def mask_tokens(self, batch_tokens):
        """make a masked input and label from batch_tokens.

        Args:
            batch_tokens (tensor): tensor of batch tokens
            batch_converter (tensor): batch converter from ESM-2.

        Returns:
            inputs: inputs with masked
            labels: labels for masked tokens
        """
        ## mask tokens
        inputs = batch_tokens.clone().to(batch_tokens.device)
        labels = batch_tokens.clone().to(batch_tokens.device)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [self.get_special_tokens_mask(val) for val in labels]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        ## Generate valid random tokens (excluding special tokens)
        valid_tokens = torch.tensor([i for i in range(self.vocab_size) if i not in self.special_token_indices])
        random_indices = torch.randint(0, len(valid_tokens), labels.shape)
        random_words = valid_tokens[random_indices].to(batch_tokens.device)  # Select only from valid tokens

        # random_words = torch.randint(self.vocab_size,
        #                             labels.shape, dtype=torch.long).to(batch_tokens.device)

        # print(indices_random)
        # print(random_words)
        # print(inputs)

        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class SimCLR(nn.Module):
    def __init__(self, model_seq, model_struct, configs):
        super(SimCLR, self).__init__()
        self.model_seq = model_seq
        self.model_struct = model_struct
        self.temperature = configs.train_settings.temperature
        self.n_views = configs.train_settings.n_views

    def forward(self, graph=None, batch_tokens=None, mode=False, return_embedding=False, return_logits=False):
        if mode == 'sequence':
            return self.model_seq(batch_tokens, return_embedding=return_embedding, return_logits=return_logits)
        elif mode == 'structure':
            return self.model_struct(graph, return_embedding=return_embedding)
        else:
            features_seq, residue_seq = self.model_seq(batch_tokens)
            features_struct, residue_struct = self.model_struct(graph)
            return features_struct, residue_struct, features_seq, residue_seq

    def forward_structure(self, graph):
        features_struct, residue_struct = self.model_struct(graph)
        return features_struct, residue_struct

    def forward_sequence(self, batch_tokens):
        features_seq, residue_seq = self.model_seq(batch_tokens)
        return features_seq, residue_seq


class GVP_CLASSIFICATION(nn.Module):
    def __init__(self, configs):
        super(GVP_CLASSIFICATION, self).__init__()
        self.model_struct = GVPEncoder(configs=configs,
                                       residue_inner_dim=configs.model.struct_encoder.residue_inner_dim,
                                       residue_out_dim=configs.model.residue_out_dim,
                                       protein_out_dim=configs.model.protein_out_dim,
                                       protein_inner_dim=configs.model.struct_encoder.protein_inner_dim,
                                       residue_num_projector=configs.model.residue_num_projector,
                                       protein_num_projector=configs.model.protein_num_projector,
                                       seqlen=configs.model.esm_encoder.max_length)

        dim_mlp = configs.model.struct_encoder.node_h_dim[0]

        # frozen all projectors of model_struct
        for name, param in self.model_struct.projectors_protein.named_parameters():
            param.requires_grad = False

        for name, param in self.model_struct.projectors_residue.named_parameters():
            param.requires_grad = False

        self.classification_layer = MoBYMLP(in_dim=dim_mlp, inner_dim=configs.model.struct_encoder.class_inner_dim,
                                            out_dim=configs.model.class_out_dim,
                                            num_layers=configs.model.class_num_projector)

    def forward(self, graph=None, mode='classification', return_embedding=False, esm2_representation=None):
        """
        graph: torch_geometric batchdasta
        mode can be structure or classification
        """
        if mode == 'structure':
            return self.model_struct(graph, return_embedding=return_embedding)
        elif mode == 'classification':
            nodes = (graph.node_s, graph.node_v)
            edges = (graph.edge_s, graph.edge_v)
            if self.model_struct.use_seq and self.model_struct.seq_embed_mode != "ESM2":
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=graph.seq)
            elif self.model_struct.use_seq and self.model_struct.seq_embed_mode == "ESM2":
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=esm2_representation
                                                                       )
            else:
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=None)

            # graph_feature=scatter_mean(residue_feature, batch, dim=0)
            graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph.batch)
            class_logits = self.classification_layer(graph_feature_embedding)
            return class_logits, graph_feature_embedding


class GVP_CONTRASTIVE_LEARNING(nn.Module):
    def __init__(self, configs):
        super(GVP_CONTRASTIVE_LEARNING, self).__init__()
        self.model_struct = GVPEncoder(configs=configs,
                                       residue_inner_dim=configs.model.struct_encoder.residue_inner_dim,
                                       residue_out_dim=configs.model.residue_out_dim,
                                       protein_out_dim=configs.model.protein_out_dim,
                                       protein_inner_dim=configs.model.struct_encoder.protein_inner_dim,
                                       residue_num_projector=configs.model.residue_num_projector,
                                       protein_num_projector=configs.model.protein_num_projector,
                                       seqlen=configs.model.esm_encoder.max_length)

        if configs.model.struct_encoder.use_seq.enable is True and \
                configs.model.struct_encoder.use_seq.seq_embed_mode == "ESM2":
            self.esm2model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.batch_converter = self.alphabet.get_batch_converter(
                truncation_seq_length=configs.model.esm_encoder.max_length)

        dim_mlp = configs.model.struct_encoder.node_h_dim[0]
        self.max_length = configs.model.esm_encoder.max_length
        # frozen all projectors of model_struct
        for name, param in self.model_struct.projectors_protein.named_parameters():
            param.requires_grad = False

        for name, param in self.model_struct.projectors_residue.named_parameters():
            param.requires_grad = False

        # self.classification_layer = MoBYMLP(in_dim=dim_mlp, inner_dim=configs.model.struct_encoder.class_inner_dim,
        #                                    out_dim=configs.model.class_out_dim,
        #                                    num_layers=configs.model.class_num_projector)

        # Yichuan: projection head for contrastive learning
        # self.projection_head = LayerNormNet(configs)
        self.batch_size = configs.train_settings.batch_size
        self.n_pos = configs.model.contrastive_learning.n_pos
        self.n_neg = configs.model.contrastive_learning.n_neg
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=dim_mlp)

        # Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_mlp,
                                                   nhead=configs.model.struct_encoder.transformer_num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=configs.model.struct_encoder.transformer_num_layers)

    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding

    @staticmethod
    def separate_features(batched_features, batch):
        # Get the number of nodes in each graph
        node_counts = batch.bincount().tolist()
        # Split the features tensor into separate tensors for each graph
        features_list = torch.split(batched_features, node_counts)
        return features_list

    def merge_features(self, features_list):
        # Pad tensors
        padded_tensors = []
        for t in features_list:
            # print("seqlength="+str(t.size(0)))
            if t.size(0) < self.max_length:
                size_diff = self.max_length - t.size(0)
                pad = torch.zeros(size_diff, t.size(1), device=t.device)
                t_padded = torch.cat([t, pad], dim=0)
            else:
                t_padded = t
            padded_tensors.append(t_padded.unsqueeze(0))  # Add an extra dimension for concatenation

        # Concatenate tensors
        result = torch.cat(padded_tensors, dim=0)
        return result

    def reorganize_batch(self, graph_feature_embedding, n_pos=None, n_neg=None):
        """
        [bsz * n_all, out_dim] -> [bsz, n_all, out_dim]
        """
        # batch_size = self.batch_size
        if n_pos is not None and n_neg is not None:
            n_all = 1
        else:
            n_all = 1 + self.n_pos + self.n_neg

        real_batch_size = graph_feature_embedding.size(0) // n_all

        # if graph_feature_embedding.size(0) != batch_size * n_all:
        #     raise ValueError(f"Input tensor size {graph_feature_embedding.size(0)} does not match expected dimensions {batch_size * n_all} (batch_size={batch_size}, n_all={n_all}).")

        if graph_feature_embedding.size(0) % n_all != 0:
            raise ValueError(
                f"Input tensor size {graph_feature_embedding.size(0)} is not divisible by {n_all}. There might be an unexpected error in the data preparation pipeline.")

        reorganized_list = []

        for i in range(real_batch_size):
            start_idx = i * n_all
            end_idx = start_idx + n_all
            reorganized_list.append(graph_feature_embedding[start_idx:end_idx])

        reorganized_graph_feature_embedding = torch.stack(reorganized_list)

        # raise ValueError(reorganized_graph_feature_embedding.shape)

        return reorganized_graph_feature_embedding

    def forward(self, graph=None, mode='classification', return_embedding=False, batch_tokens=None, n_pos=None,
                n_neg=None):
        """
        graph: torch_geometric batchdasta
        mode can be structure or classification
        """
        if mode == 'structure':
            return self.model_struct(graph, return_embedding=return_embedding)
        elif mode == 'classification':
            nodes = (graph.node_s, graph.node_v)
            edges = (graph.edge_s, graph.edge_v)
            if self.model_struct.use_seq and self.model_struct.seq_embed_mode != "ESM2":
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=graph.seq)
            elif self.model_struct.use_seq and self.model_struct.seq_embed_mode == "ESM2":
                # to do add esm2_representation
                with torch.no_grad():
                    esm2_representation = \
                    self.esm2model(batch_tokens, repr_layers=[self.esm2model.num_layers], return_contacts=False) \
                        ['representations'][self.esm2model.num_layers]
                    mask = ((batch_tokens != self.alphabet.padding_idx) & (batch_tokens != self.alphabet.cls_idx) & (
                            batch_tokens != self.alphabet.eos_idx))  # use this in v2 training

                    # print(esm2_representation)
                    # print(esm2_representation.shape)
                    esm2_representation = esm2_representation[mask]

                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=esm2_representation
                                                                       # should be`torch.Tensor` of shape [num_nodes]
                                                                       )
            else:
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=None)

            # graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph.batch)
            # projection_head = self.projection_head(self.reorganize_batch(graph_feature_embedding))
            # add transformer layer need to move inside gvp model later
            x = self.separate_features(residue_feature_embedding, graph.batch)
            # print("seperate_features")
            x = self.merge_features(x)
            # print("merge_features")
            # print(x.shape)
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            graph_feature_embedding = torch.mean(x, dim=1)  # return  [bsz * n_all, out_dim]
            projection_head = self.reorganize_batch(graph_feature_embedding, n_pos, n_neg)
            # raise ValueError(projection_head.shape)

            return projection_head


# to test the clean contrastive learning strategy.
# use this to test clean and inference_mode
# to do if GVP+ESM2 not good test this
class ESM2_CONTRASTIVE_LEARNING(nn.Module):
    def __init__(self, configs):
        super(ESM2_CONTRASTIVE_LEARNING, self).__init__()
        self.esm2model = esm.pretrained.esm2_t33_650M_UR50D()
        dim_mlp = 1280
        self.max_length = configs.model.esm_encoder.max_length
        # frozen all projectors of model_struct
        for name, param in self.esm2model.named_parameters():
            param.requires_grad = False

        self.classification_layer = MoBYMLP(in_dim=dim_mlp, inner_dim=configs.model.struct_encoder.class_inner_dim,
                                            out_dim=configs.model.class_out_dim,
                                            num_layers=configs.model.class_num_projector)

        # Yichuan: projection head for contrastive learning
        # self.projection_head = LayerNormNet(configs)
        self.batch_size = configs.train_settings.batch_size
        self.n_pos = configs.model.contrastive_learning.n_pos
        self.n_neg = configs.model.contrastive_learning.n_neg
        # Positional encoding
        # self.positional_encoding = PositionalEncoding(d_model=dim_mlp)

        # Transformer Encoder layer
        # encoder_layer = nn.TransformerEncoderLayer(d_model=dim_mlp, nhead=configs.model.struct_encoder.transformer_num_heads)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.model.struct_encoder.transformer_num_layers)

    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding

    @staticmethod
    def separate_features(batched_features, batch):
        # Get the number of nodes in each graph
        node_counts = batch.bincount().tolist()
        # Split the features tensor into separate tensors for each graph
        features_list = torch.split(batched_features, node_counts)
        return features_list

    def merge_features(self, features_list):
        # Pad tensors
        padded_tensors = []
        for t in features_list:
            # print("seqlength="+str(t.size(0)))
            if t.size(0) < self.max_length:
                size_diff = self.max_length - t.size(0)
                pad = torch.zeros(size_diff, t.size(1), device=t.device)
                t_padded = torch.cat([t, pad], dim=0)
            else:
                t_padded = t
            padded_tensors.append(t_padded.unsqueeze(0))  # Add an extra dimension for concatenation

        # Concatenate tensors
        result = torch.cat(padded_tensors, dim=0)
        return result

    def reorganize_batch(self, graph_feature_embedding, n_pos=None, n_neg=None):
        """
        [bsz * n_all, out_dim] -> [bsz, n_all, out_dim]
        """
        # batch_size = self.batch_size
        if n_pos is not None and n_neg is not None:
            n_all = 1
        else:
            n_all = 1 + self.n_pos + self.n_neg

        real_batch_size = graph_feature_embedding.size(0) // n_all

        # if graph_feature_embedding.size(0) != batch_size * n_all:
        #     raise ValueError(f"Input tensor size {graph_feature_embedding.size(0)} does not match expected dimensions {batch_size * n_all} (batch_size={batch_size}, n_all={n_all}).")

        if graph_feature_embedding.size(0) % n_all != 0:
            raise ValueError(
                f"Input tensor size {graph_feature_embedding.size(0)} is not divisible by {n_all}. There might be an unexpected error in the data preparation pipeline.")

        reorganized_list = []

        for i in range(real_batch_size):
            start_idx = i * n_all
            end_idx = start_idx + n_all
            reorganized_list.append(graph_feature_embedding[start_idx:end_idx])

        reorganized_graph_feature_embedding = torch.stack(reorganized_list)

        # raise ValueError(reorganized_graph_feature_embedding.shape)

        return reorganized_graph_feature_embedding

    def forward(self, graph=None, mode='classification', return_embedding=False, batch_tokens=None, n_pos=None,
                n_neg=None):
        """
        graph: torch_geometric batchdasta
        mode can be structure or classification
        """
        if mode == 'structure':
            return self.model_struct(graph, return_embedding=return_embedding)
        elif mode == 'classification':
            nodes = (graph.node_s, graph.node_v)
            edges = (graph.edge_s, graph.edge_v)
            with torch.no_grad():
                esm2_representation = self.esm2model(batch_tokens, repr_layers=[self.esm2model.num_layers],
                                                     return_contacts=False)
                mask = ((batch_tokens != self.esm2model.alphabet.padding_idx) & (
                            batch_tokens != self.esm2model.alphabet.cls_idx) & (
                                batch_tokens != self.esm2model.alphabet.eos_idx))  # use this in v2 training
                esm2_representation = esm2_representation[mask]

            # graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph.batch)
            # projection_head = self.projection_head(self.reorganize_batch(graph_feature_embedding))
            # add transformer layer need to move inside gvp model later
            x = self.separate_features(residue_feature_embedding, graph.batch)
            # print("seperate_features")
            x = self.merge_features(x)
            # print("merge_features")
            # print(x.shape)
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            graph_feature_embedding = torch.mean(x, dim=1)  # return  [bsz * n_all, out_dim]
            projection_head = self.reorganize_batch(graph_feature_embedding, n_pos, n_neg)
            # raise ValueError(projection_head.shape)

            return projection_head


class GVP_PRETRAIN(nn.Module):
    def __init__(self, configs):
        super(GVP_PRETRAIN, self).__init__()
        self.model_struct = GVPEncoder(configs=configs,
                                       residue_inner_dim=configs.model.struct_encoder.residue_inner_dim,
                                       residue_out_dim=configs.model.residue_out_dim,
                                       protein_out_dim=configs.model.protein_out_dim,
                                       protein_inner_dim=configs.model.struct_encoder.protein_inner_dim,
                                       residue_num_projector=configs.model.residue_num_projector,
                                       protein_num_projector=configs.model.protein_num_projector,
                                       seqlen=configs.model.esm_encoder.max_length)

        self.max_length = configs.model.esm_encoder.max_length
        self.decode_mode = configs.model.struct_decoder.mode
        dim_mlp = configs.model.struct_encoder.node_h_dim[0]
        self.transformer_after_decoder = configs.model.struct_decoder.transformer_after_decoder
        if configs.model.struct_decoder.transformer_num_layers > 0:
            self.transformer_enable = True
        else:
            self.transformer_enable = False
        # frozen all projectors of model_struct
        for name, param in self.model_struct.projectors_protein.named_parameters():
            param.requires_grad = False

        for name, param in self.model_struct.projectors_residue.named_parameters():
            param.requires_grad = False

        if self.transformer_after_decoder and self.decode_mode == "struct":
            trasnsformer_input_dim = 12  # should not be good for number of head = 4 but can still try!
        else:
            trasnsformer_input_dim = dim_mlp

        # Transformer Decoder
        if self.transformer_enable:  # do we need trasnsformer decoder??
            self.pos_embed_decoder = nn.Parameter(torch.randn(1, self.max_length, trasnsformer_input_dim) * .02)
            self.decoder_layer = nn.TransformerEncoderLayer(
                d_model=trasnsformer_input_dim,
                nhead=configs.model.struct_decoder.transformer_num_heads,
                dim_feedforward=configs.model.struct_decoder.transformer_dim_feedforward,
                activation=configs.model.struct_decoder.transformer_actfun
            )
            self.transformer_layer = nn.TransformerEncoder(self.decoder_layer,
                                                           num_layers=configs.model.struct_decoder.transformer_num_layers)

        if self.decode_mode == "struct" or self.decode_mode == "struct_gvp":  # decode as 3D structure first
            self.struct_decoder = MoBYMLP(in_dim=dim_mlp, inner_dim=configs.model.struct_decoder.inner_dim,
                                          out_dim=3 * 4, num_layers=configs.model.struct_decoder.num_layers)
        elif self.decode_mode == "gvp":
            self.struct_decoder = MoBYMLP(in_dim=dim_mlp, inner_dim=configs.model.struct_decoder.inner_dim,
                                          out_dim=dim_mlp, num_layers=configs.model.struct_decoder.num_layers)

    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding

    @staticmethod
    def separate_features(batched_features, batch):
        # Get the number of nodes in each graph
        node_counts = batch.bincount().tolist()
        # Split the features tensor into separate tensors for each graph
        features_list = torch.split(batched_features, node_counts)
        return features_list

    def merge_features(self, features_list):
        # Pad tensors
        padded_tensors = []
        for t in features_list:
            if t.size(0) < self.max_length:
                size_diff = self.max_length - t.size(0)
                pad = torch.zeros(size_diff, t.size(1), device=t.device)
                t_padded = torch.cat([t, pad], dim=0)
            else:
                t_padded = t
            padded_tensors.append(t_padded.unsqueeze(0))  # Add an extra dimension for concatenation

        # Concatenate tensors
        result = torch.cat(padded_tensors, dim=0)
        return result

    def forward(self, graph=None, mode='classification', return_embedding=False, esm2_representation=None):
        """
        graph: torch_geometric batchdasta
        mode can be structure or classification
        """
        if mode == 'structure':
            return self.model_struct(graph, return_embedding=return_embedding)
        elif mode == 'classification':
            nodes = (graph.node_s, graph.node_v)
            edges = (graph.edge_s, graph.edge_v)
            if self.model_struct.use_seq and self.model_struct.seq_embed_mode != "ESM2":
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=graph.seq)
            elif self.model_struct.use_seq and self.model_struct.seq_embed_mode == "ESM2":
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=esm2_representation
                                                                       )
            else:
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=None)

            if self.transformer_after_decoder:
                x = self.struct_decoder(residue_feature_embedding)
                x = self.separate_features(x, graph.batch)
                x = self.merge_features(x)
                if self.transformer_enable:
                    x = self.drop_positional_encoding(x, self.pos_embed_decoder)
                    x = self.transformer_layer(x)
            else:
                x = self.separate_features(residue_feature_embedding, graph.batch)
                x = self.merge_features(x)
                # print("merge shape") #[64,512,100]
                # print(x.shape)
                if self.transformer_enable:
                    x = self.drop_positional_encoding(x, self.pos_embed_decoder)
                    x = self.transformer_layer(x)

                # print("x shape") #[64,512,100]
                residue_dim = x.shape  # B,L,H
                x = self.struct_decoder(x.view(-1, residue_dim[-1])).view(residue_dim[0], residue_dim[1], -1)
                # print(x.shape) #torch.Size([128, 512, 100])

            if self.decode_mode == "struct_gvp":
                x = x  # to do

            return x


def info_nce_loss(features_struct, features_seq, n_views, temperature):  # loss function
    batch_size = len(features_struct)

    labels = torch.cat([torch.arange(batch_size, device=features_struct.device) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features_struct.device)

    features_struct = F.normalize(features_struct, dim=1)
    features_seq = F.normalize(features_seq, dim=1)
    features = torch.cat([features_struct, features_seq], dim=0)

    similarity_matrix = torch.matmul(features, features.T)

    # Discard the main diagonal from both labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features_struct.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # Select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features_struct.device)

    logits = logits / temperature
    return logits, labels


class LayerNormNet(nn.Module):
    """
    YICHUAN NEW
    From https://github.com/tttianhao/CLEAN
    """

    def __init__(self, configs):
        super(LayerNormNet, self).__init__()
        self.out_dim = configs.model.contrastive_learning.out_dim
        self.hidden_dim = configs.model.contrastive_learning.hidden_dim
        self.drop_out = configs.model.contrastive_learning.drop_out
        self.device = 'cuda'
        self.dtype = torch.float32
        feature_dim = configs.model.struct_encoder.node_h_dim[0]
        self.fc1 = nn.Linear(feature_dim, self.hidden_dim, dtype=self.dtype, device=self.device)
        self.ln1 = nn.LayerNorm(self.hidden_dim, dtype=self.dtype, device=self.device)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim,
                             dtype=self.dtype, device=self.device)
        self.ln2 = nn.LayerNorm(self.hidden_dim, dtype=self.dtype, device=self.device)
        self.fc3 = nn.Linear(self.hidden_dim, self.out_dim, dtype=self.dtype, device=self.device)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x


def print_trainable_parameters(model, logging):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_gvpclassification_model(logging, configs, accelerator):
    gvp_classification = GVP_CLASSIFICATION(configs=configs)

    if accelerator.is_main_process:
        print_trainable_parameters(gvp_classification, logging)  # 7,443,191
        print_trainable_parameters(gvp_classification.model_struct, logging)  # 616,794
        print_trainable_parameters(gvp_classification.classification_layer, logging)  # 6,826,397
        print_trainable_parameters(gvp_classification.model_struct.backbone, logging)  # 616,794

    return gvp_classification  # model_seq, model_struct #, new_model


def prepare_gvppretrain_model(logging, configs, accelerator):
    gvp_pretrain = GVP_PRETRAIN(configs=configs)

    if accelerator.is_main_process:
        print_trainable_parameters(gvp_pretrain, logging)  # 2,822,154
        print_trainable_parameters(gvp_pretrain.model_struct, logging)  # 616,794

    return gvp_pretrain  # model_seq, model_struct #, new_model


def prepare_models(configs, logging=None, accelerator=None):
    model_struct = GVPEncoder(configs=configs, residue_inner_dim=configs.model.struct_encoder.residue_inner_dim,
                              residue_out_dim=configs.model.residue_out_dim,
                              protein_out_dim=configs.model.protein_out_dim,
                              protein_inner_dim=configs.model.struct_encoder.protein_inner_dim,
                              residue_num_projector=configs.model.residue_num_projector,
                              protein_num_projector=configs.model.protein_num_projector,
                              seqlen=configs.model.esm_encoder.max_length)

    # Use ESM2 for sequence
    model_seq = ESM2(configs.model.esm_encoder.model_name,
                     residue_inner_dim=configs.model.esm_encoder.residue_inner_dim,
                     residue_out_dim=configs.model.residue_out_dim,
                     protein_out_dim=configs.model.protein_out_dim,
                     residue_num_projector=configs.model.residue_num_projector,
                     protein_num_projector=configs.model.protein_num_projector,
                     configs=configs, logging=logging)

    if hasattr(configs.train_settings, "train_lm_head_only"):
        if configs.train_settings.train_lm_head_only is True:
            for name, param in model_struct.named_parameters():
                param.requires_grad = False
            for name, param in model_seq.named_parameters():
                param.requires_grad = False
            for p in model_seq.esm2.lm_head.parameters():
                p.requires_grad = True  # will open esm2.embed_tokens.weight!

    if logging is not None and accelerator is not None and accelerator.is_main_process:
        print_trainable_parameters(model_seq, logging)
        print_trainable_parameters(model_struct, logging)

    simclr = SimCLR(model_seq, model_struct, configs=configs)
    return simclr  # model_seq, model_struct #, new_model


def prepare_gvp_contrastive_learning_model(logging, configs, accelerator):
    gvp_contrastive_learning = GVP_CONTRASTIVE_LEARNING(configs=configs)
    if accelerator.is_main_process:
        print_trainable_parameters(gvp_contrastive_learning, logging)  # 7,443,191
        print_trainable_parameters(gvp_contrastive_learning.model_struct, logging)  # 616,794
        # print_trainable_parameters(gvp_contrastive_learning.classification_layer, logging)  # 6,826,397
        print_trainable_parameters(gvp_contrastive_learning.model_struct.backbone, logging)  # 616,794
        # trainable params:  1,064,538 || all params:  2,541,722 || trainable%: 41.882550491359794
        # trainable params:  616,794 || all params:  2,093,978 || trainable%: 29.45561032637401
        # trainable params:  616,794 || all params:  616,794 || trainable%: 100.0

    return gvp_contrastive_learning  # model_seq, model_struct #, new_model


def info_nce_weights(features_struct, features_seq, device, n_views=2, mode="positive"):  # loss function
    """
    mode = "positive" only apply plddt weights on positive pairs
    mode = "all" apply plddt weights on all pairs
    """
    # print("batch_size="+str(self.args.batch_size))
    features_struct = features_struct.to(device)
    features_seq = features_seq.to(device)
    batch_size = len(features_struct)
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)],
                       dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = torch.cat([features_struct, features_seq], dim=0)
    if len(features.shape) < 2:
        features = features.unsqueeze(1)

    avg_weights_matrix = (features + features.T) / 2.0

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    avg_weights_matrix = avg_weights_matrix[~mask].view(avg_weights_matrix.shape[0], -1)
    # apply plddt weight on positives
    positives = avg_weights_matrix[labels.bool()].view(labels.shape[0], -1)
    # apply plddt weight on negatives
    # negatives = avg_weights_matrix[~labels.bool()].view(avg_weights_matrix.shape[0], -1)
    # only apply plddt weight on positives
    if mode == "positive":
        negative_weights = torch.ones(avg_weights_matrix.shape).to(device)
        negatives = negative_weights[~labels.bool()].view(negative_weights.shape[0], -1)
    elif mode == "all":
        negatives = avg_weights_matrix[~labels.bool()].view(avg_weights_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    return logits


if __name__ == '__main__':
    print('test')
