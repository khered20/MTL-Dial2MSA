import torch
from tqdm import tqdm
from .losses import combined_loss
from .utils import id_to_dialect_label
from sklearn.metrics import f1_score
import sacrebleu



def validate(model, val_loader, tokenizer, device):
    model.eval()
    translations, targets, class_preds, label_msa_ = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if 'T5' in model.__class__.__name__:
                outputs = model.translator.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=128, num_beams=3)
                class_logits = model.classifier(model.translator.encoder(batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state[:, 0, :])
            else:
                outputs = model.translator.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
#                              forced_bos_token_id=tokenizer.convert_tokens_to_ids("ar_AR"),
                                                    max_length=128, num_beams=3)
                class_logits = model.classifier(model.translator(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).encoder_last_hidden_state[:, 0, :])

            translations.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs])
            label_msa_.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['label_msa']])
            #di_sentence.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']])
            targets.extend([id_to_dialect_label(label_id) for label_id in batch["label_id"].cpu().numpy()])

            # Classification predictions
            class_preds.extend(torch.argmax(class_logits, dim=1).cpu().numpy())

    class_preds = [id_to_dialect_label(int(x)) for x in class_preds]
    bleu_score = sacrebleu.corpus_bleu(translations, [label_msa_]).score
    f1 = f1_score(targets, class_preds, average="weighted")
    return bleu_score, f1, translations, targets


def train(model, train_loader, optimizer, scheduler, device, val_loader, tokenizer,
          best_bleu=0.0, best_f1=0.0, epochs=3, eval_every=500, save_path="saved_models",alpha=0.5):
    global_step = 0
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items() if k != "label_msa"}
            batch["label_ids"] = batch.pop("label_id")

            outputs = model(**batch)
            loss = combined_loss(outputs, alpha=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            progress_bar.set_postfix(loss=loss.item())

            if global_step % eval_every == 0:
                bleu, f1, _, _ = validate(model, val_loader, tokenizer, device)
                print(f"Step {global_step}: BLEU={bleu:.4f}, F1={f1:.4f}")

                if bleu > best_bleu:
                    best_bleu = bleu
                    torch.save(model, f"{save_path}/trns/model.pt")

                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model, f"{save_path}/cls/model.pt")

        bleu, f1, _, _ = validate(model, val_loader, tokenizer, device)

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model, f"{save_path}/trns/model.pt")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model, f"{save_path}/cls/model.pt")

        print(f"dev Epoch {epoch + 1}, BLEU Score: {bleu:.4f}, F1 Score: {f1:.4f}, -best bleu: {best_bleu:.4f}, best f1: {best_f1:.4f}")

    return best_bleu, best_f1
