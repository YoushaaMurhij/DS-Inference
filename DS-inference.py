import os
import argparse
import datetime
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import train_utils
from utils import common_utils
from utils.config import cfg_from_yaml_file, log_config_to_file, global_args, global_cfg
from utils.evaluate_panoptic import init_eval, printResults
from network import build_network
from dataloader import build_dataloader

import warnings
warnings.filterwarnings("ignore")

def PolarOffsetMain(args, cfg):
    if args.launcher == None:
        dist_train = False
    else:
        args.batch_size, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.batch_size, args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True
    cfg['DIST_TRAIN'] = dist_train
    output_dir = os.path.join('./output', args.tag)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    tmp_dir = os.path.join(output_dir, 'tmp')
    summary_dir = os.path.join(output_dir, 'summary')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir, exist_ok=True)

    if args.onlyval and args.saveval:
        results_dir = os.path.join(output_dir, 'test', 'sequences')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        for i in range(8, 9):
            sub_dir = os.path.join(results_dir, str(i).zfill(2), 'predictions')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)

    if args.onlytest:
        results_dir = os.path.join(output_dir, 'test', 'sequences')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        for i in range(11,22):
            sub_dir = os.path.join(results_dir, str(i).zfill(2), 'predictions')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)

    log_file = os.path.join(output_dir, ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        total_gpus = dist.get_world_size()
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.config, output_dir))

    ### create dataloader
    if (not args.onlytest) and (not args.onlyval):
        train_dataset_loader = build_dataloader(args, cfg, split='train', logger=logger)
        val_dataset_loader = build_dataloader(args, cfg, split='val', logger=logger, no_shuffle=True, no_aug=True)
    elif args.onlyval:
        val_dataset_loader = build_dataloader(args, cfg, split='val', logger=logger, no_shuffle=True, no_aug=True)
    else:
        test_dataset_loader = build_dataloader(args, cfg, split='test', logger=logger, no_shuffle=True, no_aug=True)

    ### create model
    model = build_network(cfg)
    model.cuda()

    ### create optimizer
    optimizer = train_utils.build_optimizer(model, cfg)

    ### load ckpt
    ckpt_fname = os.path.join(ckpt_dir, args.ckpt_name)
    epoch = -1

    other_state = {}
    if args.pretrained_ckpt is not None and os.path.exists(ckpt_fname):
        #logger.info("Now in pretrain mode and loading ckpt: {}".format(ckpt_fname))
        if not args.nofix:
            if args.fix_semantic_instance:
                logger.info("Freezing backbone, semantic and instance part of the model.")
                model.fix_semantic_instance_parameters()
            else:
                logger.info("Freezing semantic and backbone part of the model.")
                model.fix_semantic_parameters()
        optimizer = train_utils.build_optimizer(model, cfg)
        epoch, other_state = train_utils.load_params_with_optimizer_otherstate(model, ckpt_fname, to_cpu=dist_train, optimizer=optimizer, logger=logger) # new feature
        logger.info("Loaded Epoch: {}".format(epoch))
    elif args.pretrained_ckpt is not None:
        train_utils.load_pretrained_model(model, args.pretrained_ckpt, to_cpu=dist_train, logger=logger)
        if not args.nofix:
            if args.fix_semantic_instance:
                logger.info("Freezing backbone, semantic and instance part of the model.")
                model.fix_semantic_instance_parameters()
            else:
                logger.info("Freezing semantic and backbone part of the model.")
                model.fix_semantic_parameters()
        else:
            logger.info("No Freeze.")
        optimizer = train_utils.build_optimizer(model, cfg)
    elif os.path.exists(ckpt_fname):
        epoch, other_state = train_utils.load_params_with_optimizer_otherstate(model, ckpt_fname, to_cpu=dist_train, optimizer=optimizer, logger=logger) # new feature
        logger.info("Loaded Epoch: {}".format(epoch))
    if other_state is None:
        other_state = {}

    ### create optimizer and scheduler
    lr_scheduler = None
    if lr_scheduler == None:
        logger.info('Not using lr scheduler')

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=True)
    logger.info(model)

    if cfg.LOCAL_RANK==0:
        writer = SummaryWriter(log_dir=summary_dir)

    logger.info('**********************Start Training**********************')
    rank = cfg.LOCAL_RANK
    best_before_iou = -1 if 'best_before_iou' not in other_state else other_state['best_before_iou']
    best_pq = -1 if 'best_pq' not in other_state else other_state['best_pq']
    best_after_iou = -1 if 'best_after_iou' not in other_state else other_state['best_after_iou']
    global_iter = 0 if 'global_iter' not in other_state else other_state['global_iter']
    val_global_iter = 0 if 'val_global_iter' not in other_state else other_state ['val_global_iter']

     
    ### evaluate
    if args.onlyval:
        logger.info('----EPOCH {} Evaluating----'.format(epoch))
        model.eval()
        min_points = 50 # according to SemanticKITTI official rule
        before_merge_evaluator = init_eval(min_points=min_points)
        after_merge_evaluator = init_eval(min_points=min_points)

        for i_iter, inputs in enumerate(val_dataset_loader):
            inputs['i_iter'] = i_iter
            torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            t = time.time()
            with torch.no_grad():
                ret_dict = model(inputs, is_test=True, before_merge_evaluator=before_merge_evaluator,
                                after_merge_evaluator=after_merge_evaluator, require_cluster=True)
                if args.saveval:
                    common_utils.save_test_results(ret_dict, results_dir, inputs)
            torch.cuda.synchronize()
            time_lag = time.time() - t
            print(time_lag)

        if rank == 0:
            ## print results
            logger.info("Before Merge Semantic Scores")
            before_merge_results = printResults(before_merge_evaluator, logger=logger, sem_only=True)
            logger.info("After Merge Panoptic Scores")
            after_merge_results = printResults(after_merge_evaluator, logger=logger)

        logger.info("----Evaluating Finished----")
        return
    

if __name__ =='__main__':
    args, cfg = global_args, global_cfg
    PolarOffsetMain(args, cfg)
