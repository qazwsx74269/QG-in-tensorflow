CUDA_VISIBLE_DEVICES=0 python  main3.py  generation --BEAM-SIZE 3 --REPLACE-UNK False --BATCH-SIZE 1 --MODE test
CUDA_VISIBLE_DEVICES=0 python  main3.py  evaluation --BEAM-SIZE 3 --REPLACE-UNK False
