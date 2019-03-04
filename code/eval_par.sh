CUDA_VISIBLE_DEVICES=0 python main4par.py generation --BATCH-SIZE 1 --MODE test --BEAM-SIZE 3 --REPLACE-UNK False
CUDA_VISIBLE_DEVICES=0 python main4par.py evaluation --BATCH-SIZE 1 --MODE test --BEAM-SIZE 3 --REPLACE-UNK False
