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

def train(model, train_loader, test_loader, sample_size, target_epsilon, lr=1e-4, batch_size=32, epochs=10, C=1):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch_size,
        sample_size=sample_size,
        epochs=epochs,
        max_grad_norm=C,
        target_epsilon=target_epsilon,
    )
    privacy_engine.attach(optimizer)

    for e in range(epochs):
        print(f"-----{e}^th epoch ------")
        for batch_idx, data in enumerate(train_loader):
            text = data["text"]
            labels = data["labels"]
            generated = data["generated"]
            answer = data["answer"]

            # Inputs are batch-first format, i.e., the first dimension of tensors must be batch dimension.
            # input_ids = #torch.randint(size=[batch_size, seq_len], low=0, high=100, device=device)
            # decoder_input_ids = #torch.randint(size=[batch_size, tar_seq_len], low=0, high=100, device=device)

            # print("text shape", type(text), len(text), len(text[0]), len(text[1]),len(text[3]))
            # print("gen shape", type(generated), len(generated), len(generated[0]))
            inputs = tokenizer( text , return_tensors="pt", padding=True, truncation=True ).to(device)
            labels_ids = tokenizer( generated, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            # labels_ids = model._shift_right(labels_ids)


            # Calling `.train()` is very important; otherwise underlying forward and backward hooks don't run.
            model.train()
            # `loss` is a 1-D tensor of shape (batch_size,).
            # TODO check
            outputs = model(**inputs, labels=labels_ids)
            # , labels=labels_ids)
            # (bsz, target_seq_len, |V|)
            logits = outputs.logits
            # CE(x=(bsz, |V|, target_seq_len), y=(bsz, target_seq_len)) => (bsz, target_seq_len)
            loss = F.cross_entropy(logits.permute(0, 2, 1), labels_ids, ignore_index=-100, reduction="none")
            loss = loss.mean(dim=1)
            optimizer.step(loss=loss)

        print(f"loss after {e}th epoch:",loss.item())
        acc_with_llm, acc_with_real = eval(model, test_loader)
        print(f"---- after {e}th epoch: acc align with llm: {acc_with_llm}, acc align with actual: {acc_with_real}")


class text_dataset():
    def __init__(self, text, labels, generated, answer):
        self.text = text
        self.labels = labels
        self.generated = generated
        self.answer = answer
        # self.n_classes = len(set(self.label))
        # self.vocab = [i for i in set(self.label)]

    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx):
            text_i = self.text[idx]
            labels_i = self.labels[idx]
            generated_i = self.generated[idx]
            answer_i = self.answer[idx]
            return {"text": text_i, "labels": labels_i, "generated": generated_i, "answer": answer_i}



def get_data(dataset, batch_size):
    # TODO
    # use test split
    data_labels=dataset["train"]["label"]
    data_text=dataset["train"]["text"]
    data_generated=dataset["train"]["generated"]
    data_answer=dataset["train"]["answer"]

    dls = text_dataset(data_text, data_labels, data_generated, data_answer)

    data_loader = DataLoader(dls)
    train_size = int(0.8 * len(data_loader))
    test_size = len(data_loader) - train_size
    sample_size = len(data_loader)
    #print("size",train_size,test_size,train_size+test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(data_loader.dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return sample_size, train_loader, test_loader


@torch.no_grad()
def eval(model, test_loader):
    count_llm = 0 
    count_real = 0
    all = 0
    for batch_idx, data in enumerate(test_loader):
            text = data["text"]
            labels = data["labels"]
            generated = data["generated"]
            answer = data["answer"]

            # Version 1
            input_ids = tokenizer( text , return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            outputs = model.generate(input_ids)
            #print("---out1:--- ", tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True),"----finished---\n")
            decode_v1 = tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # Version 2
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,  # disable sampling to test if batching affects output
            ).to(device)
            #print("---out2:--- ", tokenizer.batch_decode(output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True),"----finished---\n")
            decode_v2 = tokenizer.batch_decode(output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            
            decoded = decode_v2
            for i in range(len(decoded)):
                if decoded[i]==generated[i]:
                    count_llm+=1
                if decoded[i]==answer[i]:
                    count_real+=1
                all+=1

    acc_with_llm = count_llm / all
    acc_with_real = count_real / all
    #print("generated",generated)
    return acc_with_llm, acc_with_real

def original_acc(dataset, sample_size):
    data_labels=dataset["train"]["label"]
    data_text=dataset["train"]["text"]
    data_generated=dataset["train"]["generated"]
    data_answer=dataset["train"]["answer"]

    original_acc=0
    for i in range(sample_size):
        if data_generated[i]==data_answer[i]:
            original_acc+=1
    original_acc= original_acc/sample_size
    print(f"orginal LLM acc: {original_acc}\n")
    return original_acc

            
batch_size = 64
target_epsilon = 100
dataset = load_dataset('json', data_files='teacher/imdb/teacher.jsonl')
sample_size, train_loader, test_loader = get_data(dataset, batch_size)
original_acc(dataset, sample_size)

train(model, train_loader, test_loader, sample_size, target_epsilon, lr = 1e-5, batch_size = batch_size, epochs = 10, C = 10)
print("----- finish training -----")
acc_with_llm, acc_with_real = eval(model, test_loader)
print(f"acc align with llm: {acc_with_llm}, acc align with actual: {acc_with_real}")
