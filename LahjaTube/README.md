# LahjaTube

**LahjaTube: A Parallel Corpus of Dialectal Arabic (DA) YouTube Transcripts with Modern Standard Arabic (MSA) and English Translations**

---

## Features

- **Multi-dialectal Coverage:** Egyptian (EGY), Gulf (GLF), Levantine (LEV), and Maghrebi (MGR).
- **Parallel Translations:** Each sample includes the dialectal transcript, an aligned MSA translation, and an English translation.
- **Ethically Sourced:** Only videos with Creative Commons licenses were used; all personal and sensitive content is excluded.

- **Top Words Analysis:**  
  For each dialect, the repository includes:
  - The most frequent (top) words for each dialect only.
  - The top words from its corresponding MSA translations only.
  - Overlap between each DA and MSA.
  
- **[Word Analysis and Guidelines for Human Evaluation](https://github.com/khered20/MTL-Dial2MSA/tree/main/LahjaTube/Word%20Analysis%20and%20Guidelines%20for%20Human%20Evaluation.pdf)**
---
## Access and Usage

**LahjaTube is available for academic research purposes upon request.**  
To request access, please get in touch with us via email with the following information:

- **Full name**
- **Affiliation** (institution or organisation)
- **Purpose** of requesting the dataset (intended research or project)

**Contact:**  
📧 [abdullah.khered@manchester.ac.uk](mailto:abdullah.khered@manchester.ac.uk)

---

## Citation

If you use LahjaTube in your research, please cite:

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
