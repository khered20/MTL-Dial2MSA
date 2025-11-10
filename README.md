# MTL-Dial2MSA
Paper: A Multi-Task Learning Approach to Dialectal Arabic Identification and Translation to Modern Standard Arabic

## Features
- Joint training for **translation** and **classification**
- Supports AraT5 and AraBART pretrained models
- Validated on [Dial2MSA-Verified dev set](https://github.com/khered20/Dial2MSA-Verified/tree/main/dev) with BLEU (translation) + Weighted F1 (classification)
- Evaluation on [Dial2MSA-Verified testing set](https://github.com/khered20/Dial2MSA-Verified/blob/main/test.7z) using BLEU and chrF++ metrics multi-reference for translation and using Accuracy, Macro-Average F1 and Weighted-Average F1 for classification
- Inference mode for single or batch input

## Installation
```bash
git clone https://github.com/khered20/MTL-Dial2MSA.git
cd MTL-Dial2MSA
```
## Additional Corpora

 Additional Corpora we used in the training:
1. **PADIC** - Covers six Arabic cities from the Levant and Maghrebi regions.
   > Reference: [meftouh2018padic](https://sourceforge.net/projects/padic/).
2. **MADAR** -  Multilingual parallel dataset of 25 Arabic city-specific dialects and MSA.
   > Reference: [bouamor-etal-2018-madar](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/).
3. **Arabic STS** - Provides MSA, Egyptian, and Saudi dialect translations for English sentences.
   > Reference: [alsulaiman2022semantic](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0272991).
4. **Emi-NADI** - Our dataset to address the scarcity of Emirati dialect parallel corpora.
   > Reference: [khered2023Emi-NADI](https://github.com/khered20/UniManc_NADI2023_ArabicDialectToMSA_MT/blob/main/)

## Training and Evaluation 
This is an example code for model training using Multi-Task Learning, inference and evaluation [`MTLtrain.ipynb`](https://github.com/khered20/MTL-Dial2MSA/blob/main/MTLtrain.ipynb).

## [LahjaTube Dataset](https://github.com/khered20/MTL-Dial2MSA/tree/main/LahjaTube)

- **Total size:** 31,938 transcripts cover four **Dialectal Arabic (DA)** from YouTube alongside their **Modern Standard Arabic (MSA)** and **English** translations  
- **Distribution:**  
  - Egyptian (EGY): 10,279  
  - Gulf (GLF): 7,762  
  - Levantine (LEV): 7,695  
  - Maghrebi (MGR): 6,202
- **[Word Analysis and Guidelines for Human Evaluation](https://github.com/khered20/MTL-Dial2MSA/tree/main/LahjaTube/Word%20Analysis%20and%20Guidelines%20for%20Human%20Evaluation.pdf)**

## Citation

If you find this work or the provided dataset useful in your research or projects, please cite our paper:

```bib
@InProceedings{khered2025-mtl-da2msa,
  author    = {Khered, Abdullah  and  Benkhedda, Youcef  and  Batista-Navarro, Riza},
  title     = {A Multi-Task Learning Approach to Dialectal Arabic Identification and Translation to Modern Standard Arabic},
  booktitle      = {Proceedings of the First Workshop on Advancing NLP for Low-Resource Languages},
  month          = {September},
  year           = {2025},
  address        = {Varna, Bulgaria},
  publisher      = {Association for Computational Linguistics},
  pages     = {21--31},
  abstract  = {Translating Dialectal Arabic (DA) into Modern Standard Arabic (MSA) is a complex task due to the linguistic diversity and informal nature of dialects, particularly in social media texts. To improve translation quality, we propose a Multi-Task Learning (MTL) framework that combines DA-MSA translation as the primary task and dialect identification as an auxiliary task. Additionally, we introduce LahjaTube, a new corpus containing DA transcripts and corresponding MSA and English translations, covering four major Arabic dialects: Egyptian (EGY), Gulf (GLF), Levantine (LEV), and Maghrebi (MGR), collected from YouTube. We evaluate AraT5 and AraBART on the Dial2MSA-Verified dataset under Single-Task Learning (STL) and MTL setups. Our results show that adopting the MTL framework and incorporating LahjaTube into the training data improve the translation performance, leading to a BLEU score improvement of 2.65 points over baseline models.},
  url       = {https://acl-bg.org/proceedings/2025/LowResNLP%202025/pdf/2025.lowresnlp-1.4.pdf}
}
```

[https://aclanthology.org/2025.lowresnlp-1.4]: #
