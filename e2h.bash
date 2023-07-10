#!/usr/bin/env bash
# -*- coding:utf-8 -*-

task=$1

source function_code_e2h.bash

export verbose=True

for index in $(seq 1 ${run_time}); do
  main_output_dir=${model_folder}_run${index}
  easy_output_dir=${easy_model_folder}  # same for different seeds for saving time
  hard_output_dir=${hard_model_folder}_run${index}

  if [[ ${verbose} == True ]]
  then
    main_stdout_file=/dev/stdout
    main_stderr_file=/dev/stderr
    hard_stdout_file=/dev/stdout
    hard_stderr_file=/dev/stderr
    easy_stdout_file=/dev/stdout
    easy_stderr_file=/dev/stderr
    disable_tqdm=False
  else
    main_stdout_file=${main_output_dir}/main.log
    main_stderr_file=${main_output_dir}/main.err
    easy_stdout_file=${easy_output_dir}/easy.log
    easy_stderr_file=${easy_output_dir}/easy.err
    hard_stdout_file=${hard_output_dir}/hard.log
    hard_stderr_file=${hard_output_dir}/hard.err
    disable_tqdm=True
  fi

  echo "main_output_dir: " ${main_output_dir}
  echo "main_stdout_file: " ${main_stdout_file}
  echo "main_stderr_file: " ${main_stderr_file}

  echo "easy_output_dir: " ${easy_output_dir}
  echo "easy_stdout_file: " ${easy_stdout_file}
  echo "easy_stderr_file: " ${easy_stderr_file}

  echo "hard_output_dir: " ${hard_output_dir}
  echo "hard_stdout_file: " ${hard_stdout_file}
  echo "hard_stderr_file: " ${hard_stderr_file}

  if [[ ! -d ${main_output_dir} ]]
  then
    mkdir ${main_output_dir}
  else
    continue
  fi

  if [[ ! -d ${easy_output_dir} ]]
  then
    echo "Easy Stage ..."
    mkdir ${easy_output_dir}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${run_command} skill_${task}.py \
      --do_train --do_eval --do_predict ${constraint_decoding} ${fp16} \
      --stage easy \
      --report_to wandb \
      --use_fast_tokenizer=True \
      --ddp_find_unused_parameters=False \
      --predict_with_generate \
      --evaluation_strategy=${evaluation_strategy} \
      --save_strategy=${evaluation_strategy} \
      --metric_for_best_model eval_overall-F1 \
      --save_total_limit 1 \
      --load_best_model_at_end \
      --max_source_length=${easy_max_source_length:-"256"} \
      --max_prefix_length=${max_prefix_length:-"-1"} \
      --max_target_length=${max_target_length:-"256"} \
      --num_train_epochs=${easy_epoch} \
      --cache_dir=${cache_dir} \
      --task=${task_name} \
      --skills=${skills} \
      --empty_ratio=${empty_ratio} \
      --train_file=${data_folder}/train.json \
      --validation_file=${data_folder}/val.json \
      --test_file=${data_folder}/test.json \
      --record_schema=${data_folder}/record.schema \
      --per_device_train_batch_size=${batch_size} \
      --gradient_accumulation_steps=${gradient_accumulation_steps} \
      --per_device_eval_batch_size=$((batch_size * 4)) \
      --output_dir=${easy_output_dir} \
      --overwrite_output_dir \
      --model_name_or_path=${model_name} \
      --learning_rate=${lr} \
      --source_prefix="${task_name}: " \
      --lr_scheduler_type=${lr_scheduler} \
      --label_smoothing_factor=${label_smoothing} \
      --eval_steps ${eval_steps} \
      --decoding_format ${decoding_format} \
      --warmup_ratio ${warmup_ratio} \
      --preprocessing_num_workers=4 \
      --dataloader_num_workers=0 \
      --meta_negative=${negative} \
      --meta_positive_rate=${positive} \
      --skip_memory_metrics \
      --no_remove_unused_columns \
      --ordered_prompt=${ordered_prompt} \
      --save_better_checkpoint=False \
      --start_eval_step=${start_eval_step:-"0"} \
      --seed=${seed}${index} --disable_tqdm=${disable_tqdm} >${easy_stdout_file} 2>${easy_stderr_file}
  fi

  if [[ ! -d ${hard_output_dir} ]]
  then
    echo "Hard Stage ..."
    mkdir ${hard_output_dir}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${run_command} skill_${task}.py \
      --do_train --do_eval --do_predict ${constraint_decoding} ${fp16} \
      --stage hard \
      --report_to wandb \
      --use_fast_tokenizer=True \
      --ddp_find_unused_parameters=False \
      --predict_with_generate \
      --evaluation_strategy=${evaluation_strategy} \
      --save_strategy=${evaluation_strategy} \
      --metric_for_best_model eval_overall-F1 \
      --save_total_limit 1 \
      --load_best_model_at_end \
      --max_source_length=${max_source_length:-"256"} \
      --max_prefix_length=${max_prefix_length:-"-1"} \
      --max_target_length=${max_target_length:-"256"} \
      --num_train_epochs=${epoch} \
      --cache_dir=${cache_dir} \
      --task=${task_name} \
      --sent_num=${sent_num} \
      --M=${M} \
      --train_file=${data_folder}/train.json \
      --validation_file=${data_folder}/val.json \
      --test_file=${data_folder}/test.json \
      --record_schema=${data_folder}/record.schema \
      --per_device_train_batch_size=${batch_size} \
      --gradient_accumulation_steps=${gradient_accumulation_steps} \
      --per_device_eval_batch_size=$((batch_size * 4)) \
      --output_dir=${hard_output_dir} \
      --overwrite_output_dir \
      --model_name_or_path=${easy_output_dir} \
      --learning_rate=${lr} \
      --source_prefix="${task_name}: " \
      --lr_scheduler_type=${lr_scheduler} \
      --label_smoothing_factor=${label_smoothing} \
      --eval_steps ${eval_steps} \
      --decoding_format ${decoding_format} \
      --warmup_ratio ${warmup_ratio} \
      --preprocessing_num_workers=4 \
      --dataloader_num_workers=0 \
      --meta_negative=${negative} \
      --meta_positive_rate=${positive} \
      --skip_memory_metrics \
      --no_remove_unused_columns \
      --ordered_prompt=${ordered_prompt} \
      --save_better_checkpoint=False \
      --start_eval_step=${start_eval_step:-"0"} \
      --spot_noise=${spot_noise} \
      --asoc_noise=${asoc_noise} \
      --seed=${seed}${index} --disable_tqdm=${disable_tqdm} >${hard_stdout_file} 2>${hard_stderr_file}
  fi

  echo "Main Stage ..."
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${run_command} skill_${task}.py \
      --do_train --do_eval --do_predict ${constraint_decoding} ${fp16} \
      --stage main \
      --report_to wandb \
      --use_fast_tokenizer=True \
      --ddp_find_unused_parameters=False \
      --predict_with_generate \
      --evaluation_strategy=${evaluation_strategy} \
      --save_strategy=${evaluation_strategy} \
      --metric_for_best_model eval_overall-F1 \
      --save_total_limit 1 \
      --load_best_model_at_end \
      --max_source_length=${max_source_length:-"256"} \
      --max_prefix_length=${max_prefix_length:-"-1"} \
      --max_target_length=${max_target_length:-"256"} \
      --num_train_epochs=${epoch} \
      --cache_dir=${cache_dir} \
      --task=${task_name} \
      --train_file=${data_folder}/train.json \
      --validation_file=${data_folder}/val.json \
      --test_file=${data_folder}/test.json \
      --record_schema=${data_folder}/record.schema \
      --per_device_train_batch_size=${batch_size} \
      --gradient_accumulation_steps=${gradient_accumulation_steps} \
      --per_device_eval_batch_size=$((batch_size * 4)) \
      --output_dir=${main_output_dir} \
      --overwrite_output_dir \
      --model_name_or_path=${hard_output_dir} \
      --learning_rate=${lr} \
      --source_prefix="${task_name}: " \
      --lr_scheduler_type=${lr_scheduler} \
      --label_smoothing_factor=${label_smoothing} \
      --eval_steps ${eval_steps} \
      --decoding_format ${decoding_format} \
      --warmup_ratio ${warmup_ratio} \
      --preprocessing_num_workers=4 \
      --dataloader_num_workers=0 \
      --meta_negative=${negative} \
      --meta_positive_rate=${positive} \
      --skip_memory_metrics \
      --no_remove_unused_columns \
      --ordered_prompt=${ordered_prompt} \
      --save_better_checkpoint=False \
      --start_eval_step=${start_eval_step:-"0"} \
      --spot_noise=${spot_noise} \
      --asoc_noise=${asoc_noise} \
      --seed=${seed}${index} --disable_tqdm=${disable_tqdm} >${main_stdout_file} 2>${main_stderr_file}

  if [[ ${verbose} != True ]]
  then
    tail -n 200 ${main_output_dir}/main.log
  fi

  echo "Map Config" ${map_config}
  python3 scripts/sel2record.py -p ${main_output_dir} -g ${data_folder} -v -d ${decoding_format} -c ${map_config}
  python3 scripts/eval_extraction.py -p ${main_output_dir} -g ${data_folder} -w -m ${eval_match_mode:-"normal"}

  # delete all pytorch_model.bin of checkpoints for saving disk
  find ${main_output_dir}/ | grep -P "checkpoint-\d+/pytorch_model.bin" | xargs rm -rf
  find ${hard_output_dir}/ | grep -P "checkpoint-\d+/pytorch_model.bin" | xargs rm -rf
  find ${easy_output_dir}/ | grep -P "checkpoint-\d+/pytorch_model.bin" | xargs rm -rf
  # delete all optimizer.pt for saving disk
  find ${main_output_dir}/ | grep -P "optimizer.pt" | xargs rm -rf
  find ${hard_output_dir}/ | grep -P "optimizer.pt" | xargs rm -rf
  find ${easy_output_dir}/ | grep -P "optimizer.pt" | xargs rm -rf

done
