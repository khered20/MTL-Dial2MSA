import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, MBartForConditionalGeneration


class MultiTaskT5(nn.Module):
    def __init__(self, num_labels, pretrained_model):
        super(MultiTaskT5, self).__init__()
        self.translator = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.translator.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, label_ids=None):
        outputs = self.translator(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        translation_loss = outputs.loss
        encoder_outputs = outputs.encoder_last_hidden_state
        cls_output = encoder_outputs[:, 0, :]
        class_logits = self.classifier(cls_output)

        classification_loss = None
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(
                class_logits.view(-1, self.classifier.out_features), label_ids.view(-1)
            )

        return {
            "translation_loss": translation_loss,
            "classification_loss": classification_loss,
            "class_logits": class_logits,
        }


class MultiTaskMBart(nn.Module):
    def __init__(self, num_labels, pretrained_model):
        super(MultiTaskMBart, self).__init__()
        self.translator = MBartForConditionalGeneration.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.translator.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, label_ids=None):
        outputs = self.translator(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        translation_loss = outputs.loss
        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        cls_output = encoder_last_hidden_state[:, 0, :]
        class_logits = self.classifier(cls_output)

        classification_loss = None
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(
                class_logits.view(-1, self.classifier.out_features), label_ids.view(-1)
            )

        return {
            "translation_loss": translation_loss,
            "classification_loss": classification_loss,
            "class_logits": class_logits,
        }
