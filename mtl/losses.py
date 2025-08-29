def combined_loss(outputs, alpha=0.5):
    translation_loss = outputs["translation_loss"]
    classification_loss = outputs["classification_loss"]
    total_loss = alpha * translation_loss + (1 - alpha) * classification_loss
    return total_loss
