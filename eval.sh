cd lm-evaluation-harness;
TASKS=(
  # ANLI
  "anli_r1,anli_r2,anli_r3"
  # ARC
  "arc_challenge,arc_easy"
  # others
  "piqa,openbookqa,headqa,winogrande,logiqa,sciq"
  # hellaswag
  "hellaswag"
  # superglue
  "boolq,cb,cola,rte,wic,wsc,copa,multirc,record" 
  # LAMBADA
  "lambada_openai,lambada_standard"
)
for shot in 0 1 5; do
    for task in ${TASKS[@]}; do
        python main.py \
            --model hf-causal-experimental \
            --model_args pretrained=yahma/llama-7b-hf,peft=../lora-alpaca,use_accelerate=True \
            --tasks $task \
            --batch_size 64 \
            --device auto \
            --output_path results_s/alpaca-lora-vanilla/$task/$shot.json \
            --num_fewshot $shot
    done;
done;