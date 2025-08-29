import torch
from .utils import id_to_dialect_label
from transformers import MBartForConditionalGeneration


def predict(model, tokenizer, texts, device, max_length=128, num_beams=3):
    if isinstance(texts, str):
        texts = [texts]

    model.eval()
    results = []
    with torch.no_grad():
        encodings = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)

        gen_tokens = model.translator.generate(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
        )
        translations = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        if "T5" in model.__class__.__name__:
            encoder_hidden = model.translator.encoder(
                encodings["input_ids"], attention_mask=encodings["attention_mask"]
            ).last_hidden_state
        elif isinstance(model.translator, MBartForConditionalGeneration):
            encoder_hidden = model.translator(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            ).encoder_last_hidden_state
        else:
            raise TypeError(f"Unsupported model type: {type(model.translator)}")

        cls_output = encoder_hidden[:, 0, :]
        logits = model.classifier(cls_output)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        dialects = [id_to_dialect_label(int(p)) for p in preds]

        for inp, tr, di in zip(texts, translations, dialects):
            results.append({"input": inp, "translation": tr, "dialect": di})
    return results
