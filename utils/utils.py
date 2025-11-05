import torch
import torch.nn.functional as F
import os
import yaml
import shutil
import numpy as np
import logging as log
from scipy.ndimage import zoom
from box import Box
from pathlib import Path
import datetime
from timm import optim
from torch.utils.tensorboard import SummaryWriter
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import ast
# from Bio import SeqIO
import wandb


def prepare_wandb(project: str,
                  run_name: str = None,
                  config: dict = None,
                  save_code: bool = True,
                  api_key: str = None,
                  is_main_process: bool = True):
    if not is_main_process:
        return  # 非主进程直接跳过

    wandb.login(key=api_key)

    wandb.init(
        project=project,
        name=run_name,
        config=config or {},
        save_code=save_code
    )


def prepare_tensorboard(result_path):
    train_path = os.path.join(result_path, 'train')
    val_path = os.path.join(result_path, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    train_log_path = os.path.join(train_path, 'tensorboard')
    train_writer = SummaryWriter(train_log_path)

    val_log_path = os.path.join(val_path, 'tensorboard')
    val_writer = SummaryWriter(val_log_path)

    return train_writer, val_writer


def get_logging(result_path):
    logger = log.getLogger(result_path)
    logger.setLevel(log.INFO)

    fh = log.FileHandler(os.path.join(result_path, "logs.txt"))
    formatter = log.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = log.StreamHandler()
    logger.addHandler(sh)

    return logger


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def prepare_saving_dir(configs, config_file_path):
    """
    Prepare a directory for saving a training results.

    Args:
        configs: A python box object containing the configuration options.

    Returns:
        str: The path to the directory where the results will be saved.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    # Create the result directory and the checkpoint subdirectory.
    # result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    result_path = os.path.abspath(
        configs.result_path)  # on 10/10/2024 for each access and load the best_model for the same results_path!

    checkpoint_path = os.path.join(result_path, 'checkpoints')
    figures_path = os.path.join(result_path, 'figures')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(figures_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    shutil.copy(config_file_path, result_path)

    return result_path, checkpoint_path


def test_gpu_cuda():
    print('Testing gpu and cuda:')
    print('\tcuda is available:', torch.cuda.is_available())
    print('\tdevice count:', torch.cuda.device_count())
    print('\tcurrent device:', torch.cuda.current_device())
    print(f'\tdevice:', torch.cuda.device(0))
    print('\tdevice name:', torch.cuda.get_device_name(), end='\n\n')


def load_configs(config, args=None):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    # Convert the necessary values to floats.
    tree_config.optimizer.decay.min_lr_struct = float(tree_config.optimizer.decay.min_lr_struct)
    tree_config.optimizer.decay.min_lr_seq = float(tree_config.optimizer.decay.min_lr_seq)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    tree_config.train_settings.temperature = float(tree_config.train_settings.temperature)
    tree_config.optimizer.beta_1 = float(tree_config.optimizer.beta_1)
    tree_config.optimizer.beta_2 = float(tree_config.optimizer.beta_2)
    tree_config.model.esm_encoder.lora.dropout = float(tree_config.model.esm_encoder.lora.dropout)
    tree_config.model.struct_encoder.lora.dropout = float(tree_config.model.struct_encoder.lora.dropout)

    # overwrite parameters if set through commandline
    if args is not None:
        if args.result_path:
            tree_config.result_path = args.result_path

        if args.resume_path:
            tree_config.resume.resume_path = args.resume_path

        if args.num_end_adapter_layers:
            tree_config.encoder.adapter_h.num_end_adapter_layers = int(args.num_end_adapter_layers)

        if args.module_type:
            tree_config.encoder.adapter_h.module_type = args.module_type

        if args.restart_optimizer:
            tree_config.resume.restart_optimizer = args.restart_optimizer

    # set configs value to default if doesn't have the attr
    if not hasattr(tree_config.model.struct_encoder, "use_seq"):
        tree_config.model.struct_encoder.use_seq = None
        tree_config.model.struct_encoder.use_seq.enable = False
        tree_config.model.struct_encoder.use_seq.seq_embed_mode = "embedding"
        tree_config.model.struct_encoder.use_seq.seq_embed_dim = 20

    if not hasattr(tree_config.model.struct_encoder, "top_k"):
        tree_config.model.struct_encoder.top_k = 30  # default

    if not hasattr(tree_config.model.struct_encoder, "gvp_num_layers"):
        tree_config.model.struct_encoder.gvp_num_layers = 3  # default

    if not hasattr(tree_config.model.struct_encoder,
                   "use_rotary_embeddings"):  # configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_rotary_embeddings = False

    if not hasattr(tree_config.model.struct_encoder, "rotary_mode"):
        tree_config.model.struct_encoder.rotary_mode = 1

    if not hasattr(tree_config.model.struct_encoder,
                   "use_foldseek"):  # configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_foldseek = False

    if not hasattr(tree_config.model.struct_encoder,
                   "use_foldseek_vector"):  # configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_foldseek_vector = False

    if not hasattr(tree_config.model.struct_encoder, "num_rbf"):
        tree_config.model.struct_encoder.num_rbf = 16  # default

    if not hasattr(tree_config.model.struct_encoder, "num_positional_embeddings"):
        tree_config.model.struct_encoder.num_positional_embeddings = 16  # default

    if not hasattr(tree_config.model.struct_encoder, "node_h_dim"):
        tree_config.model.struct_encoder.node_h_dim = (100, 32)  # default
    else:
        tree_config.model.struct_encoder.node_h_dim = ast.literal_eval(tree_config.model.struct_encoder.node_h_dim)

    if not hasattr(tree_config.model.struct_encoder, "edge_h_dim"):
        tree_config.model.struct_encoder.edge_h_dim = (32, 1)  # default
    else:
        tree_config.model.struct_encoder.edge_h_dim = ast.literal_eval(tree_config.model.struct_encoder.edge_h_dim)

    return tree_config


def load_checkpoints(simclr, configs, optimizer_seq, optimizer_struct, scheduler_seq, scheduler_struct,
                     logging, resume_path, restart_optimizer=False):
    start_step = 0
    best_score = None
    assert os.path.exists(resume_path), (f'resume_path not exits')
    checkpoint = torch.load(resume_path)  # , map_location=lambda storage, loc: storage)
    print(f"load checkpoints from {resume_path}")
    logging.info(f"load checkpoints from {resume_path}")

    if "state_dict2" in checkpoint:
        simclr.model_struct.load_state_dict(checkpoint['state_dict2'])

    if "classification_layer" in checkpoint and hasattr(simclr, 'classification_layer'):
        simclr.classification_layer.load_state_dict(checkpoint['classification_layer'])

    if "struct_decoder" in checkpoint and hasattr(simclr, 'struct_decoder'):
        simclr.struct_decoder.load_state_dict(checkpoint['struct_decoder'])

    if "transformer_layer" in checkpoint and hasattr(simclr, 'transformer_layer'):
        simclr.transformer_layer.load_state_dict(checkpoint['transformer_layer'])

    if "pos_embed_decoder" in checkpoint and hasattr(simclr, 'pos_embed_decoder'):
        simclr.pos_embed_decoder.load(checkpoint['pos_embed_decoder'])

    if "state_dict1" in checkpoint:
        # to load old checkpoints that saved adapter_layer_dict as adapter_layer.
        from collections import OrderedDict
        if np.sum(["adapter_layer_dict" in key for key in checkpoint[
            'state_dict1'].keys()]) == 0:  # using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
            new_ordered_dict = OrderedDict()
            for key, value in checkpoint['state_dict1'].items():
                if "adapter_layer_dict" not in key:
                    new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                    new_ordered_dict[new_key] = value
                else:
                    new_ordered_dict[key] = value

            simclr.model_seq.load_state_dict(new_ordered_dict)
        else:  # new checkpoints with new code, that can be loaded directly.
            simclr.model_seq.load_state_dict(checkpoint['state_dict1'])

    if 'step' in checkpoint:
        if not restart_optimizer:
            if 'optimizer_struct' in checkpoint and "scheduler_struct" in checkpoint:
                optimizer_struct.load_state_dict(checkpoint['optimizer_struct'])
                logging.info('optimizer_struct is loaded to resume training!')
                scheduler_struct.load_state_dict(checkpoint['scheduler_struct'])
                logging.info('scheduler_struct is loaded to resume training!')
            if 'optimizer_seq' in checkpoint and 'scheduler_seq' in checkpoint:
                optimizer_seq.load_state_dict(checkpoint['optimizer_seq'])
                logging.info('optimizer_seq is loaded to resume training!')
                scheduler_seq.load_state_dict(checkpoint['scheduler_seq'])
                logging.info('scheduler_seq is loaded to resume training!')

            start_step = checkpoint['step'] + 1
            if 'best_score' in checkpoint:
                best_score = checkpoint['best_score']

    if hasattr(configs.model, 'memory_banck'):
        if configs.model.memory_banck.enable:
            # """
            if "seq_queue" in checkpoint:
                simclr.register_buffer('seq_queue', checkpoint['seq_queue'])
                simclr.register_buffer('seq_queue_ptr', checkpoint['seq_queue_ptr'])
                simclr.register_buffer('struct_queue', checkpoint['struct_queue'])
                simclr.register_buffer('struct_queue_ptr', checkpoint['struct_queue_ptr'])
                # simclr.seq_queue=simclr.seq_queue.to('cpu')
                # simclr.seq_queue_ptr=simclr.seq_queue_ptr.to('cpu')
                # simclr.struct_queue=simclr.struct_queue.to('cpu')
                # simclr.struct_queue_ptr=simclr.struct_queue_ptr.to('cpu')
                """
            simclr.seq_queue.load_state_dict(checkpoint['seq_queue'])
            simclr.seq_queue_ptr.load_state_dict(checkpoint['seq_queue_ptr'])
            simclr.struct_queue.load_state_dict(checkpoint['struct_queue'])
            simclr.struct_queue_ptr.load_state_dict(checkpoint['struct_queue_ptr'])
            """

    return simclr, start_step, best_score


def load_checkpoints_only(checkpoint_path, model):
    model_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict1' in model_checkpoint:
        # to load old checkpoints that saved adapter_layer_dict as adapter_layer.
        from collections import OrderedDict
        if np.sum(["adapter_layer_dict" in key for key in model_checkpoint[
            'state_dict1'].keys()]) == 0:  # using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
            new_ordered_dict = OrderedDict()
            for key, value in model_checkpoint['state_dict1'].items():
                if "adapter_layer_dict" not in key:
                    new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                    new_ordered_dict[new_key] = value
                else:
                    new_ordered_dict[key] = value

            model.load_state_dict(new_ordered_dict, strict=True)
        else:  # new checkpoints with new code, that can be loaded directly.
            model.load_state_dict(model_checkpoint['state_dict1'], strict=False)
    elif 'model_state_dict' in model_checkpoint:
        model.load_state_dict(model_checkpoint['model_state_dict'], strict=False)


def load_struct_checkpoints(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    if 'state_dict2' in checkpoint:
        state_dict = checkpoint['state_dict2']
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'dummy_param' not in k}
        model.model_struct.load_state_dict(filtered_state_dict, strict=False)
    else:
        raise (f"structure's states are not in checkpoint, please check if the correct checkpoint is loaded")


def save_checkpoints(optimizer_struct, optimizer_seq, result_path, simclr, n_steps, logging, epoch):
    checkpoint_name = f'checkpoint_{n_steps:07d}.pth'
    save_path = os.path.join(result_path, 'checkpoints', checkpoint_name)

    save_checkpoint({
        'epoch': epoch,
        'step': n_steps,
        'state_dict1': simclr.model_seq.state_dict(),
        'state_dict2': simclr.model_struct.state_dict(),
        'optimizer_struct': optimizer_struct.state_dict(),
        'optimizer_seq': optimizer_seq.state_dict(),
    }, is_best=False, filename=save_path)
    logging.info(f"Model checkpoint and metadata have been saved at {save_path}")


def save_struct_checkpoints(optimizer_struct, result_path, model, n_steps, logging, epoch, checkpoint_name=None):
    if checkpoint_name is None:
        checkpoint_name = f'checkpoint_{n_steps:07d}.pth'

    save_path = os.path.join(result_path, 'checkpoints', checkpoint_name)

    save_checkpoint({
        'epoch': epoch,
        'step': n_steps,
        'state_dict2': model.model_struct.state_dict(),
        'classification_layer': model.classification_layer.state_dict(),
        'optimizer_struct': optimizer_struct.state_dict(),
    }, is_best=False, filename=save_path)
    logging.info(f"Model checkpoint and metadata for structure have been saved at {save_path}")


def save_contrastive_struct_checkpoints(optimizer_struct, scheduler_struct, result_path, model, n_steps, logging, epoch,
                                        best_score, checkpoint_name=None):
    if checkpoint_name is None:
        checkpoint_name = f'checkpoint_{n_steps:07d}.pth'

    save_path = os.path.join(result_path, 'checkpoints', checkpoint_name)

    save_checkpoint({
        'epoch': epoch,
        'step': n_steps,
        'best_score': best_score,
        'state_dict2': model.model_struct.state_dict(),
        'wholemodel': model.state_dict(),
        'optimizer_struct': optimizer_struct.state_dict(),
        'scheduler_struct': scheduler_struct.state_dict(),
    }, is_best=False, filename=save_path)
    logging.info(f"Model checkpoint and metadata for structure have been saved at {save_path}")
    return save_path


def save_structpretrain_checkpoints(optimizer_struct, result_path, model, n_steps, logging, epoch):
    checkpoint_name = f'checkpoint_{n_steps:07d}.pth'
    save_path = os.path.join(result_path, 'checkpoints', checkpoint_name)

    save_checkpoint({
        'epoch': epoch,
        'step': n_steps,
        'state_dict2': model.model_struct.state_dict(),
        'pos_embed_decoder': model.pos_embed_decoder,
        'transformer_layer': model.transformer_layer.state_dict(),
        'struct_decoder': model.struct_decoder.state_dict(),
        'optimizer_struct': optimizer_struct.state_dict(),
    }, is_best=False, filename=save_path)
    logging.info(f"Model checkpoint and metadata for structure have been saved at {save_path}")


def prepare_optimizer(flowmodels, logging, configs):
    logging.info("prepare the optimizers")
    optimizer_seq, optimizer_struct = load_optimizers(flowmodels, logging, configs)
    scheduler_seq, scheduler_struct = None, None

    logging.info("prepare the schedulers")
    max_seq_lrs = [group['lr'] for group in optimizer_seq.param_groups]
    max_struct_lrs = [group['lr'] for group in optimizer_struct.param_groups]
    # print("max_seq_lrs:", max_seq_lrs)
    # print("Type of max_seq_lrs:", type(max_seq_lrs))
    # for i, lr in enumerate(max_seq_lrs):
    #   print(f"  max_seq_lrs[{i}]: {lr} (type: {type(lr)})")

    min_seq_lrs = [max(max_lr * float(configs.optimizer.min_lr_ratio), 1e-8) for max_lr in max_seq_lrs]
    min_struct_lrs = [max(max_lr * float(configs.optimizer.min_lr_ratio), 1e-8) for max_lr in max_struct_lrs]

    # """ #this worked #####
    scheduler_seq = CosineAnnealingWarmupRestarts(
        optimizer_seq,
        first_cycle_steps=configs.optimizer.decay.first_cycle_steps,
        cycle_mult=1.0,
        max_lr=configs.optimizer.decay.max_lr_seq,
        min_lr=configs.optimizer.decay.min_lr_seq,
        warmup_steps=configs.optimizer.decay.warmup,
        gamma=configs.optimizer.decay.gamma)
    scheduler_struct = CosineAnnealingWarmupRestarts(
        optimizer_struct,
        first_cycle_steps=configs.optimizer.decay.first_cycle_steps,
        cycle_mult=1.0,
        max_lr=configs.optimizer.decay.max_lr_struct,
        min_lr=configs.optimizer.decay.min_lr_struct,
        warmup_steps=configs.optimizer.decay.warmup,
        gamma=configs.optimizer.decay.gamma)
    # """
    # scheduler_seq=SimpleMultiGroupScheduler(optimizer_seq, max_seq_lrs, min_seq_lrs, configs)
    # scheduler_struct=SimpleMultiGroupScheduler(optimizer_seq, max_struct_lrs, min_struct_lrs, configs)
    print_scheduler_info(scheduler_seq, optimizer_seq, "Sequence", logging)
    print_scheduler_info(scheduler_struct, optimizer_struct, "Structure", logging)
    return scheduler_seq, scheduler_struct, optimizer_seq, optimizer_struct


class SingleGroupOptimizer():
    """单组optimizer包装器"""

    def __init__(self, original_optimizer, group_index):
        self.param_groups = [original_optimizer.param_groups[group_index]]

    def step(self):
        pass

    def zero_grad(self):
        pass


class SimpleMultiGroupScheduler:
    """简化的多组scheduler，只控制max_lr和min_lr"""

    def __init__(self, optimizer, max_lrs, min_lrs, configs):
        self.optimizer = optimizer
        self.individual_schedulers = []

        # 为每个参数组创建scheduler
        for i, (group, max_lr, min_lr) in enumerate(zip(optimizer.param_groups, max_lrs, min_lrs)):
            # 创建单组optimizer包装器
            single_group_opt = SingleGroupOptimizer(optimizer, i)

            # 创建scheduler，其他参数都一样
            scheduler = CosineAnnealingWarmupRestarts(
                single_group_opt,
                first_cycle_steps=configs.optimizer.decay.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=max_lr,
                min_lr=min_lr,
                warmup_steps=configs.optimizer.decay.warmup,
                gamma=configs.optimizer.decay.gamma
            )

            self.individual_schedulers.append({
                'scheduler': scheduler,
                'group_index': i
            })

    def step(self):
        """更新所有参数组的学习率"""
        for scheduler_info in self.individual_schedulers:
            scheduler = scheduler_info['scheduler']
            group_index = scheduler_info['group_index']

            # 计算新学习率并更新
            scheduler.step()
            new_lr = scheduler.get_last_lr()[0]
            self.optimizer.param_groups[group_index]['lr'] = new_lr

    def get_last_lr(self):
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


def load_optimizers(flowmodels, logging, configs):
    optimizer_seq = None
    optimizer_struct = None
    # === Sequence optimizer ===
    seq_params = []
    # ESM2 (极小学习率，因为已预训练)
    seq_params.append({
        'params': flowmodels.byol_seq2gvp.online_encoder.net.parameters(),  # ESM2
        'lr': configs.optimizer.seq_lr.esm2_encoder,  # 1e-6,
        'name': 'esm2_encoder'
    })

    # Sequence projector (ESM2输出后的projector)
    seq_params.append({
        'params': flowmodels.byol_seq2gvp.online_encoder.projector_r.parameters(),
        'lr': configs.optimizer.seq_lr.projector_r,  # 1e-4,
        'name': 'seq_projector_r'
    })

    seq_params.append({
        'params': flowmodels.byol_seq2gvp.online_encoder.projector_p.parameters(),
        'lr': configs.optimizer.seq_lr.projector_p,  # 1e-4,
        'name': 'seq_projector_p'
    })

    # Sequence相关的predictors和flow networks
    seq_params.append({
        'params': flowmodels.byol_seq2gvp.online_predictor_r.parameters(),
        'lr': configs.optimizer.seq_lr.projector_r,  # 1e-4,
        'name': 'seq2struct_predictor_r'
    })

    seq_params.append({
        'params': flowmodels.byol_seq2gvp.online_predictor_p.parameters(),
        'lr': configs.optimizer.seq_lr.projector_p,  # 1e-4,
        'name': 'seq2struct_predictor_p'
    })

    # Flow networks for seq2struct
    if hasattr(flowmodels.byol_seq2gvp, 'flow_net_r'):
        seq_params.append({
            'params': flowmodels.byol_seq2gvp.flow_net_r.parameters(),
            'lr': configs.optimizer.seq_lr.flow_net_r,  # 1e-4,
            'name': 'seq2struct_flow_r'
        })

    if hasattr(flowmodels.byol_seq2gvp, 'flow_net_p'):
        seq_params.append({
            'params': flowmodels.byol_seq2gvp.flow_net_p.parameters(),
            'lr': configs.optimizer.seq_lr.flow_net_p,  # 1e-4,
            'name': 'seq2struct_flow_p'
        })

    # 如果有seq2seq的独立predictor
    if hasattr(flowmodels, 'byol_seq2seq') and flowmodels.byol_seq2seq is not None:
        if not configs.train_settings.byol.same_predictor:
            seq_params.append({
                'params': flowmodels.byol_seq2seq.online_predictor_r.parameters(),
                'lr': configs.optimizer.seq_lr.projector_r,  # 1e-4,
                'name': 'seq2seq_predictor_r'
            })

            seq_params.append({
                'params': flowmodels.byol_seq2seq.online_predictor_p.parameters(),
                'lr': configs.optimizer.seq_lr.projector_p,  ##1e-4,
                'name': 'seq2seq_predictor_p'
            })

    # === Structure optimizer ===
    struct_params = []
    gvp_valid_params = []  # because modified gvp may has invalid param
    for name, param in flowmodels.byol_gvp2seq.online_encoder.net.named_parameters():
        if param.numel() > 0:
            gvp_valid_params.append(param)
        else:
            # Detach them from the computation graph if they're not needed
            param.requires_grad = False

    # GVP encoder
    struct_params.append({
        'params': gvp_valid_params,  # GVP
        'lr': configs.optimizer.struct_lr.gvp_encoder,  # 1e-5,
        'name': 'gvp_encoder'
    })

    # Structure projector (GVP输出后的projector)
    struct_params.append({
        'params': flowmodels.byol_gvp2seq.online_encoder.projector_r.parameters(),
        'lr': configs.optimizer.struct_lr.projector_r,  # 1e-4,
        'name': 'struct_projector_r'
    })

    struct_params.append({
        'params': flowmodels.byol_gvp2seq.online_encoder.projector_p.parameters(),
        'lr': configs.optimizer.struct_lr.projector_p,  # 1e-4,
        'name': 'struct_projector_p'
    })

    # Structure相关的predictors和flow networks
    struct_params.append({
        'params': flowmodels.byol_gvp2seq.online_predictor_r.parameters(),
        'lr': configs.optimizer.struct_lr.projector_r,  # 1e-4,
        'name': 'struct2seq_predictor_r'
    })

    struct_params.append({
        'params': flowmodels.byol_gvp2seq.online_predictor_p.parameters(),
        'lr': configs.optimizer.struct_lr.projector_p,  # 1e-4,
        'name': 'struct2seq_predictor_p'
    })

    # Flow networks for struct2seq
    if hasattr(flowmodels.byol_gvp2seq, 'flow_net_r'):
        struct_params.append({
            'params': flowmodels.byol_gvp2seq.flow_net_r.parameters(),
            'lr': configs.optimizer.struct_lr.flow_net_r,  # 1e-4,
            'name': 'struct2seq_flow_r'
        })

    if hasattr(flowmodels.byol_gvp2seq, 'flow_net_p'):
        struct_params.append({
            'params': flowmodels.byol_gvp2seq.flow_net_p.parameters(),
            'lr': configs.optimizer.struct_lr.flow_net_p,  # 1e-4,
            'name': 'struct2seq_flow_p'
        })

    # struct2struct的predictor和flow network
    if hasattr(flowmodels, 'byol_gvp2gvp') and flowmodels.byol_gvp2gvp is not None:
        struct_params.append({
            'params': flowmodels.byol_gvp2gvp.online_predictor_r.parameters(),
            'lr': configs.optimizer.struct_lr.projector_r,  # 1e-4,
            'name': 'struct2struct_predictor_r'
        })

        struct_params.append({
            'params': flowmodels.byol_gvp2gvp.online_predictor_p.parameters(),
            'lr': configs.optimizer.struct_lr.projector_p,  # 1e-4,
            'name': 'struct2struct_predictor_p'
        })

    # === 创建相同类型的optimizers ===
    # 从config中获取optimizer类型和通用参数
    optimizer_type = configs.optimizer.name.lower()  # 'adamw', 'sgd', 'adabelief'
    if optimizer_type == 'adabelief':
        optimizer_seq = optim.AdaBelief(seq_params, eps=configs.optimizer.eps, weight_decouple=True,
                                        weight_decay=configs.optimizer.weight_decay, rectify=False)
        optimizer_struct = optim.AdaBelief(struct_params, eps=configs.optimizer.eps, weight_decouple=True,
                                           weight_decay=configs.optimizer.weight_decay, rectify=False)
    elif optimizer_type == 'adam':
        if configs.optimizer.use_8bit_adam:
            import bitsandbytes
            logging.info('use 8-bit adamw')
            optimizer_seq = bitsandbytes.optim.AdamW8bit(seq_params,
                                                         betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
                                                         weight_decay=float(configs.optimizer.weight_decay),
                                                         eps=float(configs.optimizer.eps),
                                                         )
            optimizer_struct = bitsandbytes.optim.AdamW8bit(struct_params,
                                                            betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
                                                            weight_decay=float(configs.optimizer.weight_decay),
                                                            eps=float(configs.optimizer.eps),
                                                            )
        else:
            optimizer_seq = torch.optim.AdamW(seq_params,
                                              betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
                                              weight_decay=float(configs.optimizer.weight_decay),
                                              eps=float(configs.optimizer.eps)
                                              )
            optimizer_struct = torch.optim.AdamW(struct_params,
                                                 betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
                                                 weight_decay=float(configs.optimizer.weight_decay),
                                                 eps=float(configs.optimizer.eps)
                                                 )
    elif optimizer_type == 'sgd':
        logging.info('use sgd optimizer')
        optimizer_struct = torch.optim.SGD(struct_params, momentum=0.9, dampening=0,
                                           weight_decay=float(configs.optimizer.weight_decay))
        optimizer_seq = torch.optim.SGD(seq_params, momentum=0.9, dampening=0,
                                        weight_decay=float(configs.optimizer.weight_decay))

    else:
        raise ValueError('wrong optimizer')

    print_optimizer_params(optimizer_seq, "Sequence", logging)
    print_optimizer_params(optimizer_struct, "Structure", logging)
    return optimizer_seq, optimizer_struct


def print_optimizer_params(optimizer, name, logging):
    """打印optimizer参数用于验证"""
    logging.info(f"\n=== {name} Optimizer ===")
    logging.info(f"Type: {type(optimizer).__name__}")

    # 打印第一个参数组的配置（除了params）
    first_group = optimizer.param_groups[0]
    logging.info("Parameters:")
    for key, value in first_group.items():
        if key != 'params':
            logging.info(f"  {key}: {value}")


def print_scheduler_info(scheduler, optimizer, name, logging):
    """打印scheduler信息用于验证"""
    logging.info(f"\n=== {name} Scheduler Info ===")
    logging.info(f"Parameter groups: {len(optimizer.param_groups)}")

    for i, group in enumerate(optimizer.param_groups):
        group_name = group.get('name', f'group_{i}')
        current_lr = group['lr']

        # 如果scheduler有max_lr属性
        if hasattr(scheduler, 'max_lrs'):
            max_lr = scheduler.max_lrs[i] if isinstance(scheduler.max_lrs, list) else scheduler.max_lrs
            min_lr = scheduler.min_lrs[i] if isinstance(scheduler.min_lrs, list) else scheduler.min_lrs
        else:
            max_lr = "N/A"
            min_lr = "N/A"

        logging.info(f"  {group_name}: current_lr={current_lr:.2e}, max_lr={max_lr}, min_lr={min_lr}")


def plot_learning_rate(scheduler, optimizer, num_steps, filename):
    import matplotlib.pyplot as plt
    # Collect learning rates
    lrs = []

    # Simulate a training loop
    for epoch in range(num_steps):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # Plot the learning rates
    plt.plot(lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    # plt.show()
    plt.savefig(filename)


def Image_resize(matrix, max_len):
    bsz = len(matrix)
    new_size = [max_len, max_len]
    pad_contact = np.zeros((bsz, max_len, max_len), dtype=float)
    for index in range(bsz):
        scale_factors = (new_size[0] / matrix[index].shape[0], new_size[1] / matrix[index].shape[1])
        pad_contact[index] = zoom(matrix[index], zoom=scale_factors,
                                  order=0)  # order=0 for nearest-neighbor, order=3 for bicubic order=1 to perform bilinear interpolation, which smoothly scales the matrix content to the desired size.

    return pad_contact


def pad_concatmap(matrix, max_len, pad_value=255):
    bsz = len(matrix)
    pad_contact = np.zeros((bsz, max_len, max_len), dtype=float)
    mask_matrix = np.full((bsz, max_len, max_len), False, dtype=bool)
    for i in range(bsz):
        leng = len(matrix[i])
        if leng >= max_len:
            # print(i)
            pad_contact[i, :, :] = matrix[i][:max_len,
                                   :max_len]  # we trim the contact map 2D matrix if it's dimension > max_len
            mask_matrix[i, :, :] = True
        else:
            # print(i)
            # print("len < 224")
            pad_len = max_len - leng
            pad_contact[i, :, :] = np.pad(matrix[i], [(0, pad_len), (0, pad_len)], mode='constant',
                                          constant_values=pad_value)
            mask_matrix[i, :leng, :leng] = True
            # print(mask_matrix[i].shape)
            # print(mask_matrix[i])

    return pad_contact, mask_matrix


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def residue_batch_sample_old(residue_struct, residue_seq, plddt_residue, sample_size):
    sample_size = np.min([len(residue_struct), sample_size])
    random_indices = np.random.choice(len(residue_struct), sample_size, replace=False)
    residue_struct = residue_struct[random_indices, :]
    residue_seq = residue_seq[random_indices, :]
    plddt_residue = plddt_residue[random_indices]
    return residue_struct, residue_seq, plddt_residue


def residue_batch_sample(residue_struct, residue_seq, plddt_residue, sample_size, accelerator):
    # Adjust sample size if it's larger than the number of residues
    sample_size = min(residue_struct.size(0), sample_size)

    # Randomly select indices without replacement
    random_indices = torch.randperm(len(residue_struct), device=accelerator.device)[:sample_size]

    # Index the tensors to get the sampled batch
    residue_struct = residue_struct[random_indices, :]
    residue_seq = residue_seq[random_indices, :]
    plddt_residue = plddt_residue[random_indices]

    return residue_struct, residue_seq, plddt_residue


def print_gpu_memory_usage(logging):
    """Print current GPU memory usage."""
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    cached_memory = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert bytes to GB
    print(f"Allocated memory: {allocated_memory:.2f} GB")
    print(f"Cached memory: {cached_memory:.2f} GB")
    logging.info(f"Allocated memory: {allocated_memory:.2f} GB")
    logging.info(f"Cached memory: {cached_memory:.2f} GB")
