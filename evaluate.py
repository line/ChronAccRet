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

from datasets.datasets import TextMotionDataset
from models.models import ChronTMR
from models.metrics import all_contrastive_metrics, print_latex_metrics
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

def val_sim_matrix(cfg, model, test_dataloader, device, tokenizer, sentence_tokenizer, sentence_text_model, batch_size=32):

    with torch.no_grad():

        # by batch (can be too costly on cuda device otherwise)
        latent_texts = []
        latent_motions = []
        sent_embs = []
        for batch in tqdm(test_dataloader):
            
            texts, motion, length, event, shuffled_events, _ = batch
            length = length.to(device)
            motion= motion.to(device)
            
            #event = event.to(device)
            texts_token, t_length = token_process(cfg.model.token_num,cfg.model.text_model_name, texts, tokenizer)
            texts_token=texts_token.to(device)

            t_length= t_length.to(device)
            sentences = sentence_process(texts, device, sentence_tokenizer, sentence_text_model).to(device)
            
            
            texts_emb = model.text_model(texts_token, t_length).float()
            # Encode both motion and text
            latent_text,_ = model.encode(texts_emb, t_length, "txt", sample_mean=cfg.text_encoder.vae, return_distribution=cfg.text_encoder.vae)
            latent_motion,_ = model.encode(motion, length, "motion", sample_mean=cfg.text_encoder.vae, return_distribution=cfg.text_encoder.vae)

            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sentences)

        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)
        sent_embs = torch.cat(sent_embs).squeeze(1)
        
        sim_matrix = get_sim_matrix(latent_texts, latent_motions)
        
        metrics = all_contrastive_metrics(sim_matrix.cpu().numpy())
        print(metrics)
        
    return metrics["t2m/R01"], metrics["m2t/R01"]


def compute_sim_matrix(cfg, model, dataset, keyids, device, tokenizer, sentence_tokenizer, sentence_text_model, batch_size=256):

    with torch.no_grad():

        # by batch (can be too costly on cuda device otherwise)
        latent_texts = []
        latent_motions = []
        sent_embs = []
        for keyid in tqdm(keyids):
            
            texts, motion, length, event, shuffled_events, _= dataset.load_keyid(keyid)
            
            #length = np.array([length])
            length = torch.Tensor([length]).to(device).int()#.unsqueeze(0)
        
            motion= motion.to(device).unsqueeze(0)
            texts_token, t_length = token_process(cfg.model.token_num, cfg.model.text_model_name, texts, tokenizer)
            texts_token=texts_token.to(device)

            #t_length = np.array([t_length])
            t_length= torch.Tensor([t_length]).to(device).int()#.unsqueeze(0)
            sentences = sentence_process(texts, device, sentence_tokenizer, sentence_text_model).to(device).unsqueeze(0)
            #shuffled_text, e_length = token_process(cfg.model.text_model_name, shuffled_events, tokenizer)
            
            texts_emb = model.text_model(texts_token, t_length).float()
            # Encode both motion and text
            latent_text, _ = model.encode(texts_emb, t_length, "txt", sample_mean=cfg.text_encoder.vae, return_distribution=cfg.text_encoder.vae)
            latent_motion, _ = model.encode(motion, length, "motion", sample_mean=cfg.text_encoder.vae, return_distribution=cfg.text_encoder.vae)

            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sentences)

        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)
        sent_embs = torch.cat(sent_embs).squeeze(1)
        
        sim_matrix = get_sim_matrix(latent_texts, latent_motions)
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
    protocols = ["normal", "threshold", "nsim", "guo"]
    
    
    datasets = {}
    results = {}
    
    logger.info("Evaluate")
    for protocol in protocols:
        if protocol not in datasets:
            if protocol in ["normal", "threshold", "guo"]:
                dataset = TextMotionDataset(cfg,"test")
                datasets.update(
                    {key: dataset for key in ["normal", "threshold", "guo"]}
                )
            elif protocol == "nsim":
                datasets[protocol]  = TextMotionDataset(cfg,"nsim_test")
        dataset = datasets[protocol]
        
        if protocol not in results:
            if protocol in ["normal", "threshold"]:
                res = compute_sim_matrix(
                    cfg, model, dataset, dataset.keyids, device, tokenizer, sentence_tokenizer, sentence_text_model, batch_size=batch_size
                )
                results.update({key: res for key in ["normal", "threshold"]})
            elif protocol == "nsim":
                res = compute_sim_matrix(
                    cfg, model, dataset, dataset.keyids, device, tokenizer, sentence_tokenizer, sentence_text_model, batch_size=batch_size
                )
                results[protocol] = res
            elif protocol == "guo":
                keyids = sorted(dataset.keyids)
                N = len(keyids)

                # make batches of 32
                idx = np.arange(N)
                np.random.seed(0)
                np.random.shuffle(idx)
                idx_batches = [
                    idx[32 * i : 32 * (i + 1)] for i in range(len(keyids) // 32)
                ]

                # split into batches of 32
                # batched_keyids = [ [32], [32], [...]]
                results["guo"] = [
                    compute_sim_matrix(
                        cfg,
                        model,
                        dataset,
                        np.array(keyids)[idx_batch],
                        device,
                        tokenizer,
                        sentence_tokenizer,
                        sentence_text_model,
                        batch_size=batch_size,
                    )
                    for idx_batch in idx_batches
                ]
        result = results[protocol]
        # Compute the metrics
        if protocol == "guo":
            all_metrics = []
            for x in result:
                sim_matrix = x["sim_matrix"]
                metrics = all_contrastive_metrics(sim_matrix, rounding=None)
                all_metrics.append(metrics)

            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = round(
                    float(np.mean([metrics[key] for metrics in all_metrics])), 2
                )

            metrics = avg_metrics
            protocol_name = protocol
        else:
            sim_matrix = result["sim_matrix"]

            protocol_name = protocol
            if protocol == "threshold":
                emb = result["sent_emb"]
                
                threshold = 0.95#threshold_val
                protocol_name = protocol + f"_{threshold}"
            else:
                emb, threshold = None, None
            metrics = all_contrastive_metrics(sim_matrix, emb, threshold=threshold)
        
        print_latex_metrics(metrics)

        metric_name = f"{protocol_name}.yaml"
        path = os.path.join(cfg.save_dir, metric_name)
        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)
        save_metric(path, metrics)

        logger.info(f"Testing done, metrics saved in:\n{path}")
            
    

        
    

if __name__ == "__main__":
    evaluate()