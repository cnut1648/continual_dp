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


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("dataset", type=str, help="name of the dataset")
    args.add_argument("model", type=str, help="name of the model")
    return args.parse_args()

@torch.no_grad()
def eval(model, dataset, tokenizer):
    prompt = DatasetTemplates(dataset)
    template = prompt.all_template_names[0]
    print(f"For {dataset}, using template {template}")
    template = prompt[template]
    dataset_ds = load_dataset(dataset)
    output_dir = Path(__file__).parent / "teacher" / dataset
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for example in tqdm(dataset_ds["train"]):
        instructed_example, answer = template.apply(example)
        input_ids = tokenizer(instructed_example, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        example["generated"] = decoded_outputs
        example["answer"] = answer
        saved.append(example)
    # write jsonline
    with open(output_dir / "teacher.jsonl", "w") as f:
        for example in saved:
            f.write(json.dumps(example) + "\n")
    


if __name__ == "__main__":
    args = parse_args()
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    config = T5Config.from_pretrained(args.model)
    # with init_empty_weights():
        # model = AutoModelForSeq2SeqLM.from_config(config)
    # weights_location = hf_hub_download(args.model, filename="pytorch_model.bin")
    # 
    # model.tie_weights()
    # model = load_checkpoint_and_dispatch(
        # model, weights_location, device_map="auto"
    # )
    # print(model.hf_device_map)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to("cuda").eval()

    eval(model, args.dataset, tokenizer)
    
