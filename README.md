# Tokenizers
This is a repository of tokenizers that create a low-dimensional latent space for DDMs. 
# Motivation
Motivated by the paper (https://arxiv.org/abs/2401.14404), particularly its emphasis on low-dimensional latent space, we intend to construct mathematically interpretable tokenizers for DDMs that operate on a latent space. 
# The importance of the tokenizer
Here are some key quotes from the paper regarding the importance of the tokenizer:  
"Surprisingly, we discover that the main critical component is a tokenizer that creates a low-dimensional latent space."  
"We discover that it is the low-dimensional latent space, rather than the tokenizer specifics, that enables a DAE to achieve good representations."  
"These comparisons show that the tokenizer and the resulting latent space are crucial for DDM/DAE to work competitively in the self-supervised learning scenario."  
# The dimension of the latent space
Here are some key quotes from the paper regarding the dimension of the latent space:  
"We show the results w.r.t the latent dimension per token…… "  
"Figure5. ……Similar to other tokenizers we study, this pixel-based tokenizer exhibits a similar trend: a relatively small dimension of the latent space is optimal."  
"Latent dimension of the tokenizer is crucial for DDM to work well in self-supervised learning."  
"Interestingly, the optimal dimension is relatively low (d is 16 or 32), even though the full dimension per patch is much higher(16×16×3=768)."  
"Interestingly, this pixel-based tokenizer exhibits a similar trend with other tokenizers we have studied, although the optimal dimension is shifted."  
"In particular, the optimal dimension is d=48, which correspond to an image size of 64 with a patch size of 16(d=768), the linear probe accuracy drops dramatically to 23.6%."  
"The critical component is a low-dimensional latent space on which noise is added."  
