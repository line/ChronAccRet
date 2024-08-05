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
from datasets.datasets import token_process

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

def compute_sim_matrix_events(cfg, model, dataset, keyids, device, tokenizer, batch_size=256):

    with torch.no_grad():

        # by batch (can be too costly on cuda device otherwise)
        sim_matrices = []
        for keyid in tqdm(keyids):
            latent_texts = []
            latent_motions = []
            texts, motion, length, event, shuffled_events, _= dataset.load_keyid(keyid)
            length = torch.Tensor([length]).to(device).int()#.unsqueeze(0)
            motion= motion.to(device).unsqueeze(0)
            texts_token, t_length = token_process(cfg.model.token_num,cfg.model.text_model_name, texts, tokenizer)
            texts_token=texts_token.to(device)

            t_length= torch.Tensor([t_length]).to(device).int()#.unsqueeze(0)
            shuffled_text, e_length = token_process(cfg.model.token_num, cfg.model.text_model_name, shuffled_events, tokenizer)
            e_length = torch.Tensor([e_length]).to(device).int()
            shuffled_text = shuffled_text.to(device)
            shuffled_text_emb = model.text_model(shuffled_text, e_length).float()
            
            texts_emb = model.text_model(texts_token, t_length).float()
            # Encode both motion and text

            if event[0] > 1:

                latent_text, _ = model.encode(texts_emb, t_length, "txt", sample_mean=cfg.text_encoder.vae, return_distribution=cfg.text_encoder.vae)
                latent_motion, _ = model.encode(motion, length, "motion", sample_mean=cfg.motion_encoder.vae, return_distribution=cfg.motion_encoder.vae)
                latent_event, _ = model.encode(shuffled_text_emb, e_length, "txt", sample_mean=cfg.text_encoder.vae, return_distribution=cfg.text_encoder.vae)
                
                latent_texts.append(latent_text)
                latent_motions.append(latent_motion)
                latent_texts.append(latent_event)
                latent_motions = torch.cat(latent_motions)
                latent_texts = torch.cat(latent_texts)
                
                sim_matrix = get_sim_matrix(latent_motions, latent_texts)

                sim_matrices.append(sim_matrix)
                
        sim_matrices = torch.cat(sim_matrices)
    returned = {
        "sim_matrix": sim_matrices.cpu().numpy()
    }
    return returned



@hydra.main(version_base=None, config_path="config", config_name="train_bert_orig")
def retrieval(cfg: DictConfig) -> None:

    seed_everything(cfg.train.seed)
    batch_size = cfg.dataloader.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Load dataset")
    test_dataset = TextMotionDataset(
        cfg,
        "test"
    )
    
    logger.info("Load model")
    model = ChronTMR(cfg,vae=True)
    model_path = os.path.join(cfg.model_save_dir, "best_model_mt.pt")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

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
    
    result = compute_sim_matrix_events(
        cfg, model, test_dataset, test_dataset.keyids, device, tokenizer, batch_size=batch_size
    )
    
    mats = result["sim_matrix"]
    ret_res = np.sum(np.greater(mats[:, 0],mats[:, 1])) / mats.shape[0]
    logger.info(
        f"CAR: {str(ret_res)}"
    )
    metrics = {}
    metrics["m2tshuf:R@1"] = float(ret_res)


    metric_name = "shuffle_event.yaml"
    path = os.path.join(cfg.save_dir, metric_name)
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    save_metric(path, metrics)

    logger.info(f"Testing done, metrics saved in:\n{path}")


if __name__ == "__main__":
    retrieval()
