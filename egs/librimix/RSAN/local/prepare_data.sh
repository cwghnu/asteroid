#!/bin/bash

storage_dir=/exhome2/chen/data
n_src=2
python_path=/home/dcase/anaconda3/bin/python

utils/parse_options.sh

if [[ $n_src -le  1 ]]
then
  changed_n_src=2
else
  changed_n_src=$n_src
fi

$python_path local/create_local_metadata.py --librimix_dir $storage_dir/Libri$changed_n_src"Mix"

$python_path local/get_text.py \
  --libridir $storage_dir/LibriSpeech \
  --split test-clean \
  --outfile data/test_annotations.csv
