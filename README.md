# continual_dp

You can clone this repo by
```shell
git clone https://github.com/cnut1648/continual_dp.git --recursive
```

## Install

This project is tested using python 3.7 with pytorch 1.13.1 using CUDA 11.6. 
The [private_transformers](https://github.com/lxuechen/private-transformers) can be installed via
```shell
pip install git+https://github.com/lxuechen/private-transformers.git
```


Rest of the requirements can be installed via `pip install -r requirements.txt`. 

## Evaluation

Use LLM Harness:
```shell
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
pip install openai pycountry pytablewriter rouge-score sacrebleu scikit-learn sqlitedict omegaconf sentencepiece
```

Unzip `lora-alpaca` and run `./eval.sh`. You might need to change `batch_size`

## Commands

Run the teacher model by 
```
python text_gen.py imdb google/flan-t5-xxl
```