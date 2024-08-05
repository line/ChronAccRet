"""
Copyright 2024 LINE Corporation
LINE Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

import logging
import hydra
from omegaconf import DictConfig
import random, os
import numpy as np
import itertools
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from datasets.datasets import TextMotionDataset
from models.models import ChronTMR
from evaluate import val_sim_matrix
from datasets.datasets import token_process, sentence_process

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)

def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    


@hydra.main(config_path="config", config_name="train_bert_orig", version_base=None)
def train(cfg: DictConfig):

    seed_everything(cfg.train.seed)
    
    print(cfg.lmd)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Load dataloaders")
    train_dataset = TextMotionDataset(
        cfg,
        "train"
    )
    
    val_dataset = TextMotionDataset(
        cfg,
        "val"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.dataloader.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.dataloader.batch_size, 
        shuffle=False, 
        num_workers=cfg.dataloader.num_workers
    )
    
    logger.info("Load model")
    model = ChronTMR(cfg,vae=True)
    model.to(device)
    
    model.train()
    params = [
        {"params": model.text_model.parameters(), "lr": cfg.train.langlr},
        {"params": model.motion_encoder.parameters(), "lr": cfg.train.lr},
        {"params": model.text_encoder.parameters(), "lr": cfg.train.lr},
        {"params": model.motion_decoder.parameters(),"lr": cfg.train.lr},
    ]
    optimizer = torch.optim.AdamW(params)
    #switch tokenizer depending on the language model
    if cfg.model.text_model_name == 'ViT-B/32':
        from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
        tokenizer = _Tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.text_model_name, TOKENIZERS_PARALLELISM=True
        )
    logger.info(
        f"Selected Language Model: {cfg.model.text_model_name}"
    )
    sentence_tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-mpnet-base-v2', TOKENIZERS_PARALLELISM=True
    )
    sentence_text_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sentence_text_model.to(device)
    sentence_text_model.eval()
    best_te_loss = 1e5
    best_t2m_r1 = 0
    best_m2t_r1 = 0
    
    logger.info("Train")
    for epoch in range(cfg.train.epochs):
        
        step = 0
        tr_loss = 0
        model.train()
        
        for batch in tqdm(train_loader, leave=False):
            step += 1
            optimizer.zero_grad()
            texts, motion, length, event, shuffled_events, _ = batch
            length = length.to(device)
            
            
            motion= motion.to(device)
            
            event = event.to(device)
            texts_token, t_length = token_process(cfg.model.token_num, cfg.model.text_model_name, texts, tokenizer)
            texts_token=texts_token.to(device)

            t_length= t_length.to(device)
            sentences = sentence_process(texts, device, sentence_tokenizer, sentence_text_model).to(device)
            shuffled_text, e_length = token_process(cfg.model.token_num,cfg.model.text_model_name, shuffled_events, tokenizer)
            
            shuffled_text = shuffled_text.to(device)
            e_length= e_length.to(device)
            total_loss = model(texts_token, motion, sentences, shuffled_text, event, length, t_length, e_length, return_loss=True)
            
            total_loss.backward()
            optimizer.step()
            tr_loss += total_loss.item()
        tr_loss /= step
        step = 0
        te_loss = 0

        #validation
        with torch.no_grad():
            model.eval()
            t2m, m2t = val_sim_matrix(cfg, model, val_loader, device, tokenizer, sentence_tokenizer, sentence_text_model, batch_size=128)

        logger.info(
            f"epoch {epoch}, tr_loss {tr_loss}, t2m_r1 {t2m}, m2t_r1 {m2t} "
        )
        #save the best epoch
        if m2t > best_m2t_r1:
            best_m2t_r1 = m2t
            best_ep = epoch
            torch.save(model.state_dict(), os.path.join(cfg.model_save_dir, "best_model_mt.pt"))
        if t2m > best_t2m_r1:
            best_t2m_r1 = t2m
            best_ep = epoch
            torch.save(model.state_dict(), os.path.join(cfg.model_save_dir, "best_model_tm.pt"))
        torch.save(model.state_dict(), os.path.join(cfg.model_save_dir, "last_model.pt"))

        
    

if __name__ == "__main__":
    train()