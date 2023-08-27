#!/bin/bash

data_dir=$HOME/workspace/data/
mkdir -p ${data_dir}
scale=$1

cd ./tools

# generate query
# ./dsqgen -input ../query_templates/workloads.lst -directory ../query_templates/ -scale 1000 -dialect dws

# generate data
./dsdgen -scale ${scale} -dir ${data_dir} -terminate n
