#!/bin/bash

id_array=(0 1)

for i in "${id_array[@]}"
do
  nohup python run_rnn.py -i $i  1> nohup/nh_experiment_${i}.out 2>&1 &
done
