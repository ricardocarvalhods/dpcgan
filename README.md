# DP-CGAN: TensorFlow 2.0 implementation

This repository contains the code for Differentially Private Conditional GANs, originally described on [Torkzadehmahani et al. 2019](http://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Torkzadehmahani_DP-CGAN_Differentially_Private_Synthetic_Data_and_Label_Generation_CVPRW_2019_paper.pdf) and further discussed in the report included in this repository.

While originally DP-CGAN was implemented in TF 1.15, as can be seen [here](https://github.com/reihaneh-torkzadehmahani/DP-CGAN), now we include an updated version on TF 2.0 implemented from scratch. We design a custom optimizer for this task and also avoid some inconsistencies found in the original TF 1.15 implementation.

We include benchmarking on the MNIST dataset that shows improved utility when we use number of microbatches equal to batch size, compared to the original DP-CGAN approach that defined number of microbatches as 1.

Finally, we also demonstrate DP-CGAN on the Thyroid disease dataset, with empirical results giving just a drop of approximately  1\% in utility, a negligible price to pay due to adding privacy.

This project was presented on the BC AI Student Showcase 2019, poster is also included in this repository.

For any questions feel free to contact me at ricardo_silva_carvalho at sfu dot ca.
