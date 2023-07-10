#!/usr/bin/env bash
# -*- coding:utf-8 -*-
hostname
python -u -c 'import torch; print(torch.__version__); print(torch.cuda.device_count())'
nvidia-smi

export exp_name="output-e2h/EXP-RC-TB"
export batch_size=32
export gradient_accumulation_steps=2
export model_name=t5-base
export data_name=relation/conll04
export task_name="meta"
export skills="first,second,thrid,fourth"
export sent_num=2
export M=1
export empty_ratio=1
export cache_dir="./cache"
export decoding_format='spotasoc'
export lr=1e-4
export run_time="3"
export seed="42"
export CUDA_VISIBLE_DEVICES=0
export lr_scheduler=linear
export label_smoothing="0"
export epoch=50
export easy_epoch=25
export eval_steps=0
export warmup_ratio=0
export constraint_decoding=''
export verbose=False
export fp16=''
export negative=-1
export positive=1
export ordered_prompt=True
export max_source_length=256  
export easy_max_source_length=256
export max_target_length=256
export spot_noise=0
export asoc_noise=0
export map_config=config/offset_map/closest_offset_en.yaml

OPTS=$(getopt -o b:d:m:i:t:k:s:l:f:n:v --long batch:,device:,model:,skills:,sent_num:,M:,empty_ratio:,data:,task:,run-time:,seed:,lr:,lr_scheduler:,label_smoothing:,epoch:,easy_epoch:,gradient_accumulation_steps:,exp_name:,format:,eval_steps:,warmup_ratio:,constraint_decoding,verbose,preprocess,fp16:,negative:,random_prompt,max_source_length:,easy_max_source_length:,max_target_length:,spot_noise:,asoc_noise:,positive:,map_config:, -n 'parse-options' -- "$@")

if [ $? != 0 ]; then
  echo "Failed parsing options." >&2
  exit 1
fi

eval set -- "$OPTS"

while true; do
  case "$1" in
  -b | --batch)
    batch_size="$2"
    shift
    shift
    ;;
  -d | --device)
    CUDA_VISIBLE_DEVICES="$2"
    shift
    shift
    ;;
  -m | --model)
    model_name="$2"
    shift
    shift
    ;;
  -i | --data)
    data_name="$2"
    shift
    shift
    ;;
  -t | --task)
    task_name="$2"
    shift
    shift
    ;;
  -k | --run-time)
    run_time="$2"
    shift
    shift
    ;;
  -s | --seed)
    seed="$2"
    shift
    shift
    ;;
  -l | --lr)
    lr="$2"
    shift
    shift
    ;;
  -f | --format)
    decoding_format="$2"
    shift
    shift
    ;;
  -n | --negative)
    negative="$2"
    shift
    shift
    ;;
  -p | --positive)
    positive="$2"
    shift
    shift
    ;;
  --lr_scheduler)
    lr_scheduler="$2"
    shift
    shift
    ;;
  --skills)
    skills="$2"
    shift
    shift
    ;;
  --sent_num)
    sent_num="$2"
    shift
    shift
    ;;
  --M)
    M="$2"
    shift
    shift
    ;;
  --empty_ratio)
    empty_ratio="$2"
    shift
    shift
    ;;
  --label_smoothing)
    label_smoothing="$2"
    shift
    shift
    ;;
  --epoch)
    epoch="$2"
    shift
    shift
    ;;
  --easy_epoch)
    easy_epoch="$2"
    shift
    shift
    ;;
  --gradient_accumulation_steps)
    gradient_accumulation_steps="$2"
    shift
    shift
    ;;
  --exp_name)
    exp_name="$2"
    shift
    shift
    ;;
  --eval_steps)
    eval_steps="$2"
    shift
    shift
    ;;
  --warmup_ratio)
    warmup_ratio="$2"
    shift
    shift
    ;;
  --easy_max_source_length)
    easy_max_source_length="$2"
    shift
    shift
    ;;
  --max_source_length)
    max_source_length="$2"
    shift
    shift
    ;;
  --max_target_length)
    max_target_length="$2"
    shift
    shift
    ;;
  --spot_noise)
    spot_noise="$2"
    shift
    shift
    ;;
  --asoc_noise)
    asoc_noise="$2"
    shift
    shift
    ;;
  --fp16)
    fp16="$2"
    shift
    shift
    ;;
  --map_config)
    map_config="$2"
    shift
    shift
  ;;
  --constraint_decoding)
    constraint_decoding="--constraint_decoding"
    shift
    ;;
  --preprocess)
    preprocess=True
    shift
    ;;
  --random_prompt)
    ordered_prompt=False
    shift
    ;;
  -v | --verbose)
    verbose=True
    shift
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "$1" not recognize.
    exit
    ;;
  esac
done


get_gpu_num() {
  IFS=,
  num=0
  for i in ${CUDA_VISIBLE_DEVICES}
  do
    num=$((${num} + 1))
  done
  echo ${num}
  return ${num}
}

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))
}

gpu_num=$(get_gpu_num)
# 若使用多 GPU，则使用 distributed 版本的 PyTorch
# For multiple GPU, use the Distributed version of PyTorch
if [[ ${gpu_num} == 1 ]]
then
    run_command=python3
else
    master_port=$(rand 10000 50000)
    echo "Master Port: ${master_port}"
    run_command="python3 -m torch.distributed.launch --nproc_per_node ${gpu_num} --master_port ${master_port}"
fi

echo "run_command: " ${run_command}

echo "Map Config" ${map_config}

# 不指定 eval_steps 则每一个 epoch 进行一次模型验证
# Without specifying eval_steps, model validation is performed once for each epoch
if [[ ${eval_steps} == 0 ]]
then
  evaluation_strategy='epoch'
else
  evaluation_strategy='steps'
fi

# google/mt5-base -> google_mt5-base
model_name_log=$(echo ${model_name} | sed -s "s/\//_/g")
data_name_log=$(echo ${data_name} | sed -s "s/\//_/g")
batch_log=$((gpu_num * batch_size * gradient_accumulation_steps))

echo "exp_name: " ${exp_name}

model_folder=${exp_name}/main/${skills}_${data_name_log}_${model_name_log}_${decoding_format}_e${epoch}_ee${easy_epoch}_${max_source_length}_${easy_max_source_length}_${lr_scheduler}_lr${lr}_ls${label_smoothing}_b${batch_log}_wu${warmup_ratio}_er${empty_ratio}
easy_model_folder=${exp_name}/easy/${skills}_${data_name_log}_${model_name_log}_${decoding_format}_ee${easy_epoch}_${easy_max_source_length}_${lr_scheduler}_lr${lr}_ls${label_smoothing}_b${batch_log}_wu${warmup_ratio}_er${empty_ratio}
hard_model_folder=${exp_name}/hard/${skills}_${data_name_log}_${model_name_log}_${decoding_format}_e${epoch}_ee${easy_epoch}_${max_source_length}_${easy_max_source_length}_${lr_scheduler}_lr${lr}_ls${label_smoothing}_b${batch_log}_wu${warmup_ratio}_er${empty_ratio}

if [[ ${sent_num} != 1 ]]
then
  hard_model_folder=${hard_model_folder}_sn${sent_num}_m${M}
  model_folder=${model_folder}_sn${sent_num}_m${M}
fi

if [[ ! -d ${exp_name} ]]
then
  mkdir ${exp_name}
fi

if [[ ! -d ${exp_name}/easy ]]
then
  mkdir ${exp_name}/easy
fi

if [[ ! -d ${exp_name}/hard ]]
then
  mkdir ${exp_name}/hard
fi

if [[ ! -d ${exp_name}/main ]]
then
  mkdir ${exp_name}/main
fi

if [[ ${constraint_decoding} != "" ]]
then
  model_folder=${model_folder}_CD
fi

if [[ ${ordered_prompt} == False ]]
then
  model_folder=${model_folder}_RP
  hard_model_folder=${hard_model_folder}_RP
fi

if [[ ${spot_noise} != 0 ]]
then
  model_folder=${model_folder}_sn${spot_noise}
  hard_model_folder=${hard_model_folder}_sn${spot_noise}
fi

if [[ ${asoc_noise} != 0 ]]
then
  model_folder=${model_folder}_an${asoc_noise}
  hard_model_folder=${hard_model_folder}_an${asoc_noise}
fi

if [[ ${positive} != 1 ]]
then
  model_folder=${model_folder}_p${positive}
fi

data_folder=data/text2${decoding_format}/${data_name}

export TOKENIZERS_PARALLELISM=false

if [[ ${fp16} != "" ]]
then
  fp16="--fp16 --fp16_backend apex --fp16_opt_level ${fp16}"
fi

export PYTHONPATH="${PYTHONPATH}:./"
