How Redundant Is the Transformer Stack in Speech Representation Models?
==============================

The repository contains code used in the paper *"How Redundant Is the Transformer Stack in Speech Representation Models?"*, including for other domains than audio.

---

We provide code for extracting layer-wise features for several Transformer-based networks, including `wav2vec2`, `wavLM`, `dinov2`, etc.. See ```transformer_redundancy/methods``` for similarity metric, including CKA, mutual kNN and Procrustes similarity

### Experiments
- Layer-wise analyses in ```transformer_redundancy/run.py```.
- Knowledge-distillation with mimicking layers trains with ```transformer_redundancy/train_surrogate.py```.

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
