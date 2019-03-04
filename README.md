# QG-in-tensorflow
This repository only includes source code written in tensorflow because datasets are a little big.   
The data directory should have included SQUAD1.1 data and its processed data.   
The data2.0 directory should have included SQUAD2.0 data and its processed data.   
The code reimplements the idea in 《Learning to Ask: Neural Question Generation for Reading Comprehension》.

# model architecture
2 layers bi-lstm encoder + 2 layers bi-lstm decoder + lulong attention  
In predicted stage, beam search + replace UNK.  
# usage
`bash run.sh`  

If you have any questions, welcome to contact to me!