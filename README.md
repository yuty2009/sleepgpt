# SleepGPT: A Language Model Built on Sleep Stage Sequences Enables Efficient Sleep Assessment

## Overview

![Framework](https://github.com/yuty2009/sleepgpt/blob/main/figures/framework.png)
**Overview of the proposed SleepGPT model and applications to sleep staging enhancement and sleep disorder diagnosis**. (**a**) The SleepGPT model is pretrained on a large sleep stage annotation dataset and is used to correct the sleep stage predictions of existing sleep staging models. Moreover, a hierarchical transformer network (HTN) is employed for sleep disorder diagnosis, with SleepGPT acting as a local feature extractor. (**b**) Datasets for evaluating the proposed artificial intelligence (AI) models. For sleep staging, cross-validation of the AI models is performed on the SleepEDF and MASS datasets. The models trained from the MASS datasets are then externally validated on the Physio2018 dataset for generalizability assessment. Furthermore, the translatability of the SleepGPT model on wearable EEG data is validated on the BOAS dataset with simultaneously collected PSG and headband EEG data. For sleep disorder diagnosis, cross-validation of the AI models is performed on the CAP and MNC datasets. For generalizability assessment, models trained on the CAP dataset for distinguishing normal from abnormal sleep stage sequences are validated externally using the ISRUC, MNC, and HANG7 datasets, while those trained on the MNC dataset for distinguishing Type-1 narcolepsy from other hypersomnia and healthy controls are externally validated on the HANG7 dataset.

![SleepGPT](https://github.com/yuty2009/sleepgpt/blob/main/figures/sleepgpt.png)
The architecture of the proposed SleepGPT model comprises a series of $L$ transformer decoder blocks. Overlapping blocks consisting of $K$ stages are extracted from an overnight sleep stage sequence with a stride of 1. Here, the $(i-1)$th input block of $K$ sleep stages are fed into the model for predicting the $i$th target block.

![Hierarchical Transformer Network](https://github.com/yuty2009/sleepgpt/blob/main/figures/sleepgpt_htn.png)
The hierarchical transformer network (HTN) for stage sequence-based sleep disorder diagnosis. The HTN model comprises a local feature extractor, i.e., the SleepGPT model, a transformer encoder responsible for global feature extraction, and a classification head dedicated to diagnosis. The configuration of the global transformer encoder is depicted to the right. Positioned at the bottom is an example of a whole-night sleep stage sequence, partitioned into non-overlapping segments that are subsequently input into the HTN model.

## Prepare the data

Download the datasets (access requirements may apply) from the following links:

- [SHHS](https://sleepdata.org/datasets/shhs)
- [SleepEDF](https://physionet.org/content/sleep-edfx/1.0.0/)
- [MASS](http://ceams-carsm.ca/mass/)
- [CAP](https://physionet.org/content/capslpdb/1.0.0/)

Replace the data path in the code with the path to the downloaded data. Run the code in the "datasets" folder to preprocess the data. For example, to preprocess the SHHS dataset, run the following code:

```python
python datasets/shhsreader.py
```

## Run the code

### Train SleepGPT

```python
python main_train_slm.py 
```

### Sleep Staging with SleepGPT

Stage with pretrained sleep staging models

```python
python main_stage_slm.py
```

or [YASA](https://github.com/raphaelvallat/yasa/tree/master)

```python
python main_yasa_slm.py
```

### Sleep Disorder Diagnosis

```python
python main_diagnose_cv.py
```

## Citation

If you use the code or results in your research, please consider citing our work at:

```
@article{yu2024sleepgpt,
  title={A Language Model Built on Sleep Stage Sequences Enables Efficient Sleep Assessment},
  author={Yu, Tianyou and Gu, Zhenghui and Huang, Rui and Wang, Fei and Li, Man and Yu, Jingang and Yu, Zhuliang and Zhang, Jun and Xu, Yan and Jiang, Haiteng and Liu, Wenjuan and Deng, Guifeng and Gao, Zhengrun and Wu, Yiwen and Liu, Jun and Zhang, Yu and Jones, Matt W and Li, Yuanqing and Xiao, Jun and Wu, Wei},
  journal={medRxiv},
  year={2024},
  doi={10.1101/2024.10.26.24316166},
  url={https://www.medrxiv.org/content/10.1101/2024.10.26.24316166v4},
}
```

## Acknowledgements
This project uses resources from:
- [YASA](https://github.com/raphaelvallat/yasa)
- [minGPT](https://github.com/karpathy/minGPT)
- [Transformers](https://github.com/huggingface/transformers)