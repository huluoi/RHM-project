# RHM project
 This is a project that investigates the connection between network layer representations and dataset internal structure hierarchy by linear probing.
 # Motivation
 Motivated by the paper (https://arxiv.org/abs/2307.02129), particularly its emphasis on hierachical structure, we intend to study the connection between the representation of each layer of the network and the internal structure hierarchy of the dataset. 
 # Methodology
 1. Train the RHM classification task on FCN, and during the training process, extract the representation of each layer of the network on the train set and test set (V_1, V_2, V_3, V_4).  
 2. Use the representation on the train set to perform linear regression on the RHM class label, shallow hidden feature, deep hidden feature, and original data input. Thus, a linear model is obtained.  
 3. Use this linear model trained on the train set and the representation on the test set to predict the class label, shallow hidden feature, deep hidden feature, and original data input of the RHM test set. And we study which layer of the network's representation can better predict these quantities of RHM. 
 # Usage
 ```
 py main.py --device cpu --dataset hier1 --net fcn --output .\output\output.txt --num_layers 3 --net_layers 6 --epochs 1000
 ```

 
