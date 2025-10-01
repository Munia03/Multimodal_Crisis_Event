"""
@author: Chonghan Chen <chonghac@cs.cmu.edu>
"""
from os import path as osp
import os
import logging
from PIL import Image
from torch.serialization import save
from args import get_args
from trainer import Trainer
from crisismmd_dataset import CrisisMMDataset
from models import DenseNetBertMMModel_crisiskan, ImageOnlyModel, TextOnlyModel
from models_clip import CLIPDiffModel, CLIP_CrisiKAN
import os
import numpy as np
import torch
from torch.nn.modules import activation
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import math
# from optimization import AdamW, WarmupCosineSchedule , WarmupLinearSchedule
import time
from torch.optim import Adam, AdamW
import random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

import nltk
nltk.download('stopwords')
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Can slow down but ensures deterministic results




if __name__ == '__main__':
    opt = get_args()

    model_to_load = opt.model_to_load
    image_model_to_load = opt.image_model_to_load
    text_model_to_load = opt.text_model_to_load

    device = opt.device
    num_workers = opt.num_workers

    EVAL = opt.eval
    USE_TENSORBOARD = opt.use_tensorboard
    SAVE_DIR = opt.save_dir
    MODEL_NAME = opt.model_name if opt.model_name else str(int(time.time()))

    MODE = opt.mode
    TASK = opt.task
    MAX_ITER = opt.max_iter
    OUTPUT_SIZE = None 
    if TASK == 'task1':
        OUTPUT_SIZE = 2
    elif TASK == 'task2_full':
        OUTPUT_SIZE = 8
    elif TASK == 'task2':
        OUTPUT_SIZE = 6
    elif TASK == 'task3':
        OUTPUT_SIZE = 3
    else:
        raise NotImplemented

    # General hyper parameters
    learning_rate = opt.learning_rate
    batch_size = opt.batch_size

    # Create folder for saving
    save_dir = osp.join(SAVE_DIR, MODEL_NAME)
    if not osp.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not osp.exists(save_dir):
        os.mkdir(save_dir)

    # set logger
    logging.basicConfig(filename=osp.join(save_dir, 'output_{}.log'.format(int(time.time()))), level=logging.INFO)

    # writing arguments to log file
    logging.info(str(opt) + '\n')

    train_loader, dev_loader = None, None
    train_load,  dev_load, test_load = [], [], []
    if not EVAL:
        train_set = CrisisMMDataset()
        train_set.initialize(opt, phase='train', cat='all',
                                 task=TASK)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for data in tqdm(train_loader, total=len(train_loader)):
            train_load.append(data)

    dev_set = CrisisMMDataset()
    dev_set.initialize(opt, phase='dev', cat='all',
                       task=TASK)

    dev_loader = DataLoader(
        dev_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for data in tqdm(dev_loader, total=len(dev_loader)):
        dev_load.append(data)

    test_set = CrisisMMDataset()
    test_set.initialize(opt, phase='test', cat='all',
                        task=TASK)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    for data in tqdm(test_loader, total=len(test_loader)):
        test_load.append(data)

    loss_fn = nn.CrossEntropyLoss()


    if MODE == 'text_only':
        model = TextOnlyModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
    elif MODE == 'image_only':
        model = ImageOnlyModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
    elif MODE == 'both':
        # model = DenseNetBertMMModel_crisiskan(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
        model = CLIP_CrisiKAN(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
        # model = CLIPDiffModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
        
    else:
        raise NotImplemented

    model = nn.DataParallel(model)

    t_total = math.ceil((len(train_load))/(40)) * 50

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, cooldown=0, verbose=True)



    model.to(device)
    trainer = Trainer(train_load, dev_load, test_load,
                      model, loss_fn, optimizer, scheduler, eval=EVAL, device=device, tensorboard=USE_TENSORBOARD, mode=MODE)

    if model_to_load:
        model.module.load(model_to_load)
        logging.info("\n***********************")
        logging.info("Model Loaded!")
        logging.info("***********************\n")
    if text_model_to_load:
        model.module.load(text_model_to_load)
    if image_model_to_load:
        model.module.load(image_model_to_load)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable params: {trainable_params}, Total params: {total_params}")


    if not EVAL:
        logging.info("\n================Training Summary=================")
        logging.info("Training Summary: ")
        logging.info("Learning rate {}".format(learning_rate))
        logging.info("Batch size {}".format(batch_size))
        logging.info(trainer.model)
        logging.info("\n=================================================")

        trainer.train(MAX_ITER)

        logging.info("\n=================================================")
        
        set_seed(42)  # Call before loading model

        model.module.load_best()
        trainer.validate()
        trainer.predict()

    else:
        set_seed(42)  # Call before loading model

        logging.info("\n================Evaluating Model=================")
        logging.info(trainer.model)

        model.module.load_best()
        trainer.validate()
        trainer.predict()


