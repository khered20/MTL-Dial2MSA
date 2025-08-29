# MTL-Dial2MSA
Multi-Task Learning for Dialect Arabic → Modern Standard Arabic (MSA) Translation + Dialect Identification

## Features
- Joint training for **translation** and **dialect classification**
- Supports AraT5 and AraBART backbones
- Evaluation with BLEU (translation) + F1 (classification)
- Inference mode for single or batch input

## Installation
```bash
git clone https://github.com/<your-username>/MTL-Dial2MSA.git
cd MTL-Dial2MSA
pip install -r requirements.txt
