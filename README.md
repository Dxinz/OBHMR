# DeepOBHMR
OBHMR: Partial Generalized Point Set Registration with Overlap Bidirectional Hybrid Mixture Model.
The repository offers the official implementation of our paper in PyTorch.

## Abstract ##
 In this paper, we introduce a novel overlap-based bidirectional registration approach, i.e., Overlap Bidirectional Hybrid Mixture Registration (OBHMR). Our approach incorporates geometric information (i.e., normal vectors) in both the correspondence and transformation stages and formulates the optimization objective of registration in a bidirectional manner. To address the issue of partial-to-full registration, our approachutilizes the overlap score of each point to formulate the overlap-guided Hybrid Mixture Model consisting of the Gaussian Mixture Model (GMM) and Fisher Mixture Model (FMM).OBHMR contains four components: (1) the correspondencenetwork that estimates the correspondence probabilities; 
(2) the overlap prediction network that calculates the overlap score ofeach point; 
(3) the learning posterior module that formulates the overlap-guided HMM parameters; 
(4) the bidirectionaltransformation module that computes the rigid transformation by utilizing the bidirectional registration mechanism, givencorrespondenceand overlap-guided HMM parameters.
 Experiments using 291 human femur and 260 hip models show the improvement in partial-to-full registration performance (p < 0.01) under different overlapping ratios. Furthermore, individual contributions of three modules in OBHMR havebeen validated in ablation studies. The results demonstrate OBHMR’s capability of achieving accurate registration for computer-assisted orthopedic surgery.



**We will release our all code soon!**

