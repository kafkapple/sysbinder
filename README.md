# Neural Systematic Binder
*ICLR 2023*

#### [[arXiv](https://arxiv.org/abs/2211.01177)] [[project](https://sites.google.com/view/neural-systematic-binder)] [[datasets](https://drive.google.com/drive/folders/1FKEjZnKfu9KnSGfnr8oGVUBSPqnptzJc?usp=sharing)] [[openreview](https://openreview.net/forum?id=ZPHE4fht19t)]

This is the **official PyTorch implementation** of _Neural Systematic Binder_.

<img src="https://i.imgur.com/hqwcCpU.png">

### Authors
Gautam Singh and Yeongbin Kim and Sungjin Ahn

### Datasets
The datasets tested in the paper (CLEVR-Easy, CLEVR-Hard, and CLEVR-Tex) can be downloaded via this [link](https://drive.google.com/drive/folders/1FKEjZnKfu9KnSGfnr8oGVUBSPqnptzJc?usp=sharing).

#### Text Dataset Support
In addition to the image datasets, this implementation now supports text data processing using the ISEAR (International Survey on Emotion Antecedents and Reactions) dataset. The ISEAR dataset contains emotion-labeled statements that are processed using BERT embeddings.

### Training
To train the model, simply execute:
```bash
python train.py
```
Check `train.py` to see the full list of training arguments. You can use the `--data_path` argument to point to the set of images via a glob pattern.

For text data training, use:
```bash
python train.py --data_type isear --data_path data/isear/isear.csv
```

### Text Processing Parameters
The implementation includes the following text-specific parameters:
- `--data_type`: Choose between 'clevr' (image) or 'isear' (text)
- `--d_model`: Dimension of text embeddings (default: 768, matching BERT)
- `--num_slots`: Number of slots for text processing (default: 4)
- `--num_blocks`: Number of processing blocks (default: 8)
- `--slot_size`: Size of each slot (default: 2048)
- `--mlp_hidden_size`: Hidden layer size for MLP (default: 192)

### Text Embedding
- Uses BERT (`bert-base-uncased`) for text embedding
- Implements mean pooling for sentence representation
- Includes a text processor for dimension alignment
- Supports both image and text modalities in the same architecture

### Outputs
The training code produces Tensorboard logs. To see these logs, run Tensorboard on the logging directory that was provided in the training argument `--log_path`. These logs contain:
- Training loss curves
- Visualizations of reconstructions and object attention maps
- Text embedding visualizations (for text data)
- Attention patterns in text processing

### Packages Required
The following packages may need to be installed first.
- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://pypi.org/project/tensorboard/) for logging
- [Transformers](https://huggingface.co/docs/transformers/index) for BERT text embeddings
- [pandas](https://pandas.pydata.org/) for CSV processing
- [scikit-learn](https://scikit-learn.org/) for data splitting

### Evaluation
The evaluation scripts are provided in branch `evaluate`.

### Citation
```
@inproceedings{
      singh2023sysbinder,
      title={Neural Systematic Binder},
      author={Gautam Singh and Yeongbin Kim and Sungjin Ahn},
      booktitle={International Conference on Learning Representations},
      year={2023},
      url={https://openreview.net/forum?id=ZPHE4fht19t}
}
```