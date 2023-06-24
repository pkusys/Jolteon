#!/bin/bash

data_dir=$HOME/workspace/serverless-bound/TPC-DS/DSGen/data
mkdir -p ${data_dir}
repo_name=workspace/serverless-bound/TPC-DS
scale=$1

cd $HOME/${repo_name}/DSGen/tools

# generate query
# ./dsqgen -input ../query_templates/workloads.lst -directory ../query_templates/ -scale 1000 -dialect dws

# generate data
./dsdgen -scale ${scale} -dir ${data_dir} -terminate n

# num_chunks=10
# for i in $(seq 1 ${num_chunks}) 
# do
#     ./dsdgen -scale ${scale} -dir ${data_dir} -terminate n -parallel ${num_chunks} -child ${i}
# done

cd $HOME