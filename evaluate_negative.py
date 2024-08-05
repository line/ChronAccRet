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
import yaml
import hydra
from omegaconf import DictConfig
import random, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import json

from datasets.datasets import TextMotionDataset
from models.models import ChronTMR 
from models.metrics import negative_contrastive_metrics, print_latex_metrics_neg
from datasets.datasets import token_process, sentence_process

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)

def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

# TMR evaluation criterion
def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)

def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix




def compute_sim_matrix(cfg, model, dataset, keyids, device, tokenizer, sentence_tokenizer, sentence_text_model):

    with torch.no_grad():

        latent_texts = []
        latent_motions = []
        sent_embs = []
        latent_events =[]
        textsdict = {}
        indx = 0
        eventindx = 4384 
        keysdict = {}
        for keyid in tqdm(keyids):
            
            texts, motion, length, event, shuffled_events, _= dataset.load_keyid(keyid)
            textsdict[str(indx)] = texts
            keysdict[str(indx)] = str(keyid)
            indx = indx + 1
            
            length = torch.Tensor([length]).to(device).int()
        
            motion= motion.to(device).unsqueeze(0)
            texts_token, t_length = token_process(cfg.model.token_num,cfg.model.text_model_name, texts, tokenizer)
            texts_token=texts_token.to(device)

            
            t_length= torch.Tensor([t_length]).to(device).int()
            sentences = sentence_process(texts, device, sentence_tokenizer, sentence_text_model).to(device).unsqueeze(0)
            #introduce shuffled events if there is more than one event
            shuffled_text, e_length = token_process(cfg.model.token_num,cfg.model.text_model_name, shuffled_events, tokenizer)
            shuffled_text=shuffled_text.to(device)
            e_length= torch.Tensor([e_length]).to(device).int()
            texts_emb = model.text_model(texts_token, t_length).float()
            # Encode both motion and text
            latent_text, _ = model.encode(texts_emb, t_length, "txt", sample_mean=cfg.text_encoder.vae, return_distribution=cfg.text_encoder.vae)
            latent_motion, _ = model.encode(motion, length, "motion", sample_mean=cfg.motion_encoder.vae, return_distribution=cfg.motion_encoder.vae)
            if event > 1:
                textsdict[str(eventindx)] = shuffled_events
                keysdict[str(eventindx)] = str(keyid)
                eventindx = eventindx + 1
                event_emb = model.text_model(shuffled_text, e_length).float()
                latent_event, _ = model.encode(event_emb, t_length, "txt", sample_mean=cfg.text_encoder.vae, return_distribution=cfg.text_encoder.vae)
                latent_events.append(latent_event)

            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sentences)
        latent_events = torch.cat(latent_events)
        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)
        sent_embs = torch.cat(sent_embs).squeeze(1)

        #join shuffled
        latent_texts = torch.cat((latent_texts,latent_events),0)
        
        sim_matrix = get_sim_matrix(latent_texts, latent_motions)
    txt_name = "text_names.json"
    path = os.path.join(cfg.save_dir, txt_name)
    with open(path, mode="wt", encoding="utf-8") as f:
	    json.dump(textsdict, f, ensure_ascii=False, indent=2)
    key_name = "key_names.json"
    path = os.path.join(cfg.save_dir, key_name)
    with open(path, mode="wt", encoding="utf-8") as f:
	    json.dump(keysdict, f, ensure_ascii=False, indent=2)
    returned = {
        "sim_matrix": sim_matrix.cpu().numpy(),
        "sent_emb": sent_embs.cpu().numpy(),
    }
    return returned


@hydra.main(config_path="config", config_name="train_bert_orig", version_base=None)
def evaluate(cfg: DictConfig):

    seed_everything(cfg.train.seed)
    batch_size = cfg.dataloader.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Load dataset")
    test_dataset = TextMotionDataset(
        cfg,
        "test"
    )
    test_dataset_nsmi = TextMotionDataset(
        cfg,
        "nsim_test"
    )
    
    
    
    logger.info("Load model")
    model = ChronTMR(cfg,vae=True)
    model_path = os.path.join(cfg.model_save_dir, "best_model_mt.pt")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    
    
    model.eval()
    #switch tokenizer depending on the language model
    if cfg.model.text_model_name == 'ViT-B/32':
        from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
        tokenizer = _Tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.text_model_name, TOKENIZERS_PARALLELISM=False
        )
    logger.info(
        f"Selected Language Model: {cfg.model.text_model_name}"
    )
    sentence_tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-mpnet-base-v2', TOKENIZERS_PARALLELISM=False
    )
    sentence_text_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sentence_text_model.to(device)
    sentence_text_model.eval()
    protocols = ["normal"]
    
    
    datasets = {}
    results = {}
    
    logger.info("Evaluate")
    for protocol in protocols:
        if protocol not in datasets:
            
            dataset = TextMotionDataset(cfg,"test")
            datasets.update(
                {key: dataset for key in ["normal"]}
            )
        dataset = datasets[protocol]
        if protocol not in results:
            
            res = compute_sim_matrix(
                cfg, model, dataset, dataset.keyids, device, tokenizer, sentence_tokenizer, sentence_text_model
            )
            results.update({key: res for key in ["normal"]})
        # Compute the metrics
        sim_matrix = res["sim_matrix"]
        protocol_name = protocol
        emb, threshold = None, None
        metrics = negative_contrastive_metrics(sim_matrix, emb, threshold=threshold)
        
        print_latex_metrics_neg(metrics)

        metric_name = f"{protocol_name}_neg.yaml"
        path = os.path.join(cfg.save_dir, metric_name)
        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)
        save_metric(path, metrics)

        logger.info(f"Testing done, metrics saved in:\n{path}")
            
    

if __name__ == "__main__":
    evaluate()