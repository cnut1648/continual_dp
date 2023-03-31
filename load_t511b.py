import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, T5Config
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download
from promptsource.promptsource.templates import DatasetTemplates
from datasets import load_dataset
from tqdm.auto import tqdm
from pathlib import Path
import os, json
import numpy as np
import transformers
from private_transformers import PrivacyEngine
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_arch = "t5-small"
model_arch = "t5-11b"
model = transformers.T5ForConditionalGeneration.from_pretrained(model_arch).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_arch)
print("done")