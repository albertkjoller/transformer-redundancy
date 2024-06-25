Is the Transformer stack completely redundant?
==============================

Research project by Albert Kjøller Jacobsen (part of Cognitive Spaces)

Contains the following analysis methods:
- CKA similarity
- Mutual KNN similarity
- Procrustes similarity
- Jacobian similarity

along with layer-wise analysis procedures for the following Transformer networks:
- wav2vec2
- wavLM
- dinov2
- deit
- beit
- (CLIP)

Layer-wise analyses and model pruning experiments can be carried out by running the ```transformer_redundancy/run.py``` script.

Knowledge-distillation of audio models (and dinov2) can be carried out by running the ```transformer_redundancy/train_surrogate.py``` script.