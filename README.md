How Redundant Is the Transformer Stack in Speech Representation Models?
==============================

The repository contains code used in the paper *"How Redundant Is the Transformer Stack in Speech Representation Models?"*, including for other domains than audio.

---

### Experiments

We provide code for extracting layer-wise features for several Transformer-based networks, including `wav2vec2`, `wavLM`, `dinov2`, etc.. 

- Similarity metrics including CKA, mutual kNN and Procrustes similarity found in ```transformer_redundancy/methods```.
- Layer-wise analyses are performed with ```transformer_redundancy/run.py```.
- Knowledge-distillation with mimicking layers runs with ```transformer_redundancy/train_surrogate.py```.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2db60169-ea93-423b-803a-56bded7dcc6b" alt="main-result" style="width:85%;"/>
</p>

--- 

### Citation
```
@inproceedings{dorszewski2025redundant,
  title={How Redundant Is the Transformer Stack in Speech Representation Models?},
  author={Dorszewski, Teresa and Jacobsen, Albert Kj{\o}ller and T{\v{e}}tkov{\'a}, Lenka and Hansen, Lars Kai},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
