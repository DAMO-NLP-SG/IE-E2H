size=$1
dataset=$2
task=$3

if [[ ! -d output-e2h ]]
then
    mkdir output-e2h
fi

. config/data_conf/${size}_${dataset}_e2h.ini && bash run_exp_e2h.bash e2h.bash ${task}
