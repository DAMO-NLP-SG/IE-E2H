#!/bin/bash

echo "Start Running ..."

script=$1
task=$2

export device=0
export run_time="3"
export lr_scheduler=linear
export eval_steps=0
export ordered_prompt=True
export negative=-1
export map_config=config/offset_map/closest_offset_en.yaml

read -ra GRADIENT_ACCUMULATION_STEPS <<<"${GRADIENT_ACCUMULATION_STEPS}"
read -ra LR_RATE <<<"${LR_RATE}"
read -ra WARMUP_PROP <<<"${WARMUP_PROP}"
read -ra EPOCH <<<"${EPOCH}"
read -ra EASY_EPOCH <<<"${EASY_EPOCH}"
read -ra LABEL_SMOOTHING <<<"${LABEL_SMOOTHING}"
read -ra SENT_NUM <<<"${SENT_NUM}"
read -ra M <<<"${M}"
read -ra EMPTY_RATIO <<<"${EMPTY_RATIO}"
read -ra NOISE <<<"${NOISE}"
read -r model_name <<<"${model_name}"
read -r data_name <<<"${data_name}"
read -r exp_name <<<"${exp_name}"
read -r skills <<<"${skills}"
read -r max_source_length <<<"${max_source_length}"
read -r easy_max_source_length <<<"${easy_max_source_length}"
read -r batch_size <<<"${batch_size}"

echo "model_name: " ${model_name}
echo "data_name: " ${data_name}
echo "exp_name: " ${exp_name}
echo "skills: " ${skills}
echo "max_source_length: " ${max_source_length}
echo "easy_max_source_length: " ${easy_max_source_length}


for learning_rate in "${LR_RATE[@]}"; do
  echo "learning rate " ${learning_rate}
  for gradient_accumulation_steps in "${GRADIENT_ACCUMULATION_STEPS[@]}"; do
    echo "batch size " $((batch_size * gradient_accumulation_steps))
    for epoch in "${EPOCH[@]}"; do
      echo "epoch " ${epoch}
      for easy_epoch in "${EASY_EPOCH[@]}"; do
        echo "easy_epoch " ${easy_epoch}
        for empty_ratio in "${EMPTY_RATIO[@]}"; do
          echo "empty_ratio " ${empty_ratio}
          for sent_num in "${SENT_NUM[@]}"; do
            echo "sent_num " ${sent_num}
            for m in "${M[@]}"; do
              echo "m " ${m}
              for noise in "${NOISE[@]}"; do
                echo "noise " ${noise}
                for warmup_ratio in "${WARMUP_PROP[@]}"; do
                  echo "warmup ratio " ${warmup_ratio}

                    bash ${script} ${task} -k ${run_time} \
                      -m ${model_name} \
                      -i ${data_name} \
                      --lr_scheduler linear \
                      --epoch ${epoch} \
                      --device ${device} \
                      --easy_epoch ${easy_epoch} \
                      --gradient_accumulation_steps ${gradient_accumulation_steps} \
                      --exp_name ${exp_name} \
                      --skills ${skills} \
                      --eval_steps ${eval_steps} \
                      --batch ${batch_size} \
                      --sent_num ${sent_num} \
                      --M ${m} \
                      --empty_ratio ${empty_ratio} \
                      --lr ${learning_rate} \
                      --warmup_ratio ${warmup_ratio} \
                      --max_source_length ${max_source_length} \
                      --easy_max_source_length ${easy_max_source_length} \
                      --spot_noise ${noise} --asoc_noise ${noise} \
                      --negative ${negative} --map_config ${map_config}

                    bash scripts/summary_performance.bash ${exp_name} > ${exp_name}/best.performance.now

                done
              done
            done
          done
        done
      done
    done
  done
done

bash scripts/summary_performance.bash ${exp_name} > ${exp_name}/best.performance.now

exit 0
