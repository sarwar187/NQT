#!/bin/sh
echo $2
echo $1
if [ $2 == 'test' ]; then 
	CUDA_VISIBLE_DEVICES=$1 python translate_baseline.py -load_weights clir_it_en_large -src_lang it -trg_lang en -qdir data/queries/index/italian/all/ -qtitle All-Top-ita-tit2.txt -qdesc All-Top-ita-tit-desc.txt 
else
	CUDA_VISIBLE_DEVICES=$1 python train.py -src_data ../transformer-tensorflow/data/material_it_en/train.enc -trg_data ../transformer-tensorflow/data/material_it_en/train.dec -src_lang it -trg_lang en -epochs 5
fi
#echo "$1"
#python --version
#CUDA_VISIBLE_DEVICES=$1 python train.py -src_data data/french.txt -trg_data data/english.txt -src_lang fr -trg_lang en -epochs 10

