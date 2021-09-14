预测是否存在 motif

主页：http://cnn.csail.mit.edu/

论文：Zeng H., Edwards M.D., Gifford D. K.(2015) "Convolutional Neural Network Architectures for Predicting DNA-Protein Binding".
Proceedings of Intelligent Systems for Molecular Biology (ISMB) 2016
Bioinformatics, 32(12):i121-i127. doi: 10.1093/bioinformatics/btw255.

论文作者 GitHub 源码：https://github.com/gifford-lab/Keras-genomics

简介：数据集包含大量 DNA结合蛋白序列 片段，如："CAGTTGGCC...CAAAGGGAACACACAAGTAGA"，以及对应标签 1。
标签 1 代表该片段上有结合位点（称之为 motif，具体来看就是一个子串），标签 0 代表不存在。所以，这是一个
二分类任务。

# 数据集

数据集下载：http://cnn.csail.mit.edu/motif_discovery/

序列长这样：
```
>chr20:42901189-42901289 CAGTTGGCCTCATTGTGAGATCGCCTCCAGCTGGCAGACCCAGAGAAGTGCAGAAGGGGCTCATCCAGGCCTGGGGTTTGCAAAGGGAACACACAAGTAGA 1
>chr9:120407047-120407147_shuf ACGGCAACAGTTTCGGGCAGTTGCTGGTTCCCAACATCATGGTAAATTTTATGGTGGCGAGTCCAGCATGCTGTTGCTTGCACCATTGACCAGGGTAAGGT 0
```




