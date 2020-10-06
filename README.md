# DP-CGAN: TensorFlow 2.0 implementation

This repository contains the code for Differentially Private Conditional GANs, originally described on [Torkzadehmahani et al. 2019](http://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Torkzadehmahani_DP-CGAN_Differentially_Private_Synthetic_Data_and_Label_Generation_CVPRW_2019_paper.pdf) and further discussed in the report included in this repository.

While originally DP-CGAN was implemented in TF 1.15, as can be seen [here](https://github.com/reihaneh-torkzadehmahani/DP-CGAN), now we include an updated version on TF 2.0 implemented from scratch. We design a custom optimizer for this task and also avoid some inconsistencies found in the original TF 1.15 implementation.

We include benchmarking on the MNIST dataset that shows improved utility when we use number of microbatches equal to batch size, compared to the original DP-CGAN approach that defined number of microbatches as 1.

Finally, we also demonstrate DP-CGAN on the Thyroid disease dataset, with empirical results giving just a drop of approximately  1\% in utility, a negligible price to pay due to adding privacy.

This project was presented on the BC AI Student Showcase 2019, poster is also included in this repository.

For any questions feel free to contact me at ricardo_silva_carvalho at sfu dot ca.

---

Below we give a jupyter notebook containing the implementation of a Differentially Private Conditional GAN, originally described on Torkzadehmahani et al. 2019, with explanation of every step implemented. 

We include a TensorFlow 2 version implemented from scratch, using the Keras API and a tf.GradientTape training loop. Also to successfully use DP on a Conditional GAN, we design a custom optimizer. Experiments used the MNIST dataset.

**Pre-requisites**: (see links on notebook for suggestions of references)
- Generative Adversarial Networks (GANs)
- Differential Privacy (DP), especially DP-SGD

**We focus on the hypothetical scenario where**:
- Training data is considered sensitive.
- We aim to release a "safe" version (that does not compromise privacy) of the training data to the public.
- The goal is to allow external people to use the "safe" training data to create a classification model that performs well on the real testing data.
  - Therefore, we validate our GANs by creating models on the generated data and measuring performance.

Regarding the use of any GAN with DP, please consider the following notes.

**<div style='color:red'>Important notes about using DP with GANs:</div>**
- Usually the Generator is trained without DP, therefore we cannot show the data labels to the Generator. We sample labels uniformly at random on each training step.
- Although the notebook shows one training, to report results we should train it many times from scratch and show averaged results, not maximum or single result.
- Our Validation uses the training data again to evaluate the GAN. To be 100% DP, we need to either pay some privacy budget for this step, or just use the actual generated data with cross validation for evaluation.
- When generating data for creating classification models, we create the same amount of data for each label. Using the real test distribution of labels to determine the number of examples generated for each label assumes this information is public, which is rarely the case.

**Links**:
- <a target="_blank" href="https://colab.research.google.com/github/ricardocarvalhods/dpcgan/blob/master/DP_CGAN_MNIST.ipynb">Open directly on Google Colab by clicking here</a>
  - It takes 3+ hours to run the experiment completely once on Google Colab GPU
  - Taking this long is normal, due to clipping gradients per-example
- [Jupyter Notebook](https://github.com/ricardocarvalhods/dpcgan/blob/master/DP_CGAN_MNIST.ipynb)
