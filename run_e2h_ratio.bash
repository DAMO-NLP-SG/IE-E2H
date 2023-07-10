size=$1
dataset=$2
task=$3

if [[ ! -d output-e2h-ratio ]]
then
    mkdir output-e2h-ratio
fi

. config/data_conf_ratio/${size}_${dataset}_e2h.ini && bash run_exp_e2h.bash e2h_ratio.bash ${task}
