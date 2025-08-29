import gc
import torch
import sacrebleu
import subprocess
from sklearn.metrics import f1_score


def dialect_label_to_id(dialect_label):
    label_map = {"MSA": 0, "Egypt": 1, "Gulf": 2, "Levantine": 3, "Maghreb": 4}
    return label_map[dialect_label]


def id_to_dialect_label(class_id):
    id_map = {0: "MSA", 1: "Egypt", 2: "Gulf", 3: "Levantine", 4: "Maghreb"}
    return id_map.get(class_id, "Unknown")


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def calculate_bleu(dataset_name, prediction_file, python_exec="python"):
    """Optional external BLEU script call"""
    try:
        result = subprocess.run(
            [python_exec, "bleuV6_yt.py", dataset_name, prediction_file],
            capture_output=True,
            text=True,
            check=True,
        )
        bleu_score = float(result.stdout.strip())
        return bleu_score
    except Exception as e:
        print(f"BLEU calc error: {e}")
        return None


def compute_scores(preds, refs, targets, class_preds):
    bleu_score = sacrebleu.corpus_bleu(preds, [refs]).score
    f1 = f1_score(targets, class_preds, average="weighted")
    return bleu_score, f1
