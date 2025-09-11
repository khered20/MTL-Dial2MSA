# MTL-Dial2MSA
Multi-Task Learning for Dialect Arabic → Modern Standard Arabic (MSA) Translation + Dialect Identification

## Features
- Joint training for **translation** and **dialect classification**
- Supports AraT5 and AraBART backbones
- Training and validated with BLEU (translation) + Weighted F1 (classification)
- Evaluation on Dial2MSA testing set
- Inference mode for single or batch input

## Installation
```bash
git clone https://github.com/khered20/MTL-Dial2MSA.git
cd MTL-Dial2MSA
