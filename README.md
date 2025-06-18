## Federated Class Incremental Learning
Pytorch implementation of the paper "Forget Less, Learn More: Contrastive-Based Federated Class Incremental Learning with a Low-Dimensional Projection Layer"\ 
Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) 2025\
[[pdf]]

## Abstract
Federated Class Incremental Learning (FCIL) extends Federated Learning (FL) to dynamic environments where clients continually encounter new classes over time, but past data becomes inaccessible. This leads to catastrophic forgetting of previous classes. Therefore, maintaining a balance between learning new classes (plasticity) and retaining past knowledge (stability) is crucial. To address these challenges, we propose a contrastive-based FCIL framework with a low-dimensional projection layer to enhance both stability and plasticity. A low-dimensional projection layer is introduced to improve the generalization capability of the feature extractor and stability-plasticity trade-off. In this regard, we employ supervised contrastive learning in the projection layer during the initial task to strengthen the feature extractor for better adaptation to new classes. Additionally, we train a data-free generator on the server and distribute it to clients to replay synthetic samples from past tasks. To balance stability and plasticity, we design a novel loss function that integrates cross-entropy loss, feature-level knowledge distillation loss, contrastive loss in the projection layer, and classification head refinement. Extensive experiments demonstrate that our framework outperforms state-of-the-art FCIL baselines, achieving higher accuracy and lower forgetting.
## Installation

### Prerequisite
* python == 3.9
* torch == 1.13.1
* torchvision == 0.14.1

### Dataset
 * Download the datasets CIFAR-10, CIFAR-100, PPMI, VOC2012 and set the directory in --path. Additionally, set the method to FCILLD.


# Run Code

<div>
  <button class="copy-button" onclick="copyToClipboard('code-to-copy')"></button>
  <pre><code id="code-to-copy">python main.py --dataset={DATASET_NAME} --method=FCILLD --num_clients=5 --path={PATH_TO_DATASET}</code></pre>
</div>



# Citation

Please cite our paper if you find the repository helpful.

```
@inproceedings{khazaei2025forget,
  title={Forget Less, Learn More: Contrastive-Based Federated Class Incremental Learning with a Low-Dimensional Projection Layer},
  author={Khazaei, Ensieh and Hatzinakos, Dimitrios},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={1791--1800},
  year={2025}
}
```
