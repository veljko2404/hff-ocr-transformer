from config import NUM_CLASSES, DEVICE
from models.model_cnn_transformer import CNNTransformerOCR

model = CNNTransformerOCR(NUM_CLASSES).to(DEVICE)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total: {total}")
print(f"Trainable: {trainable}")

"""
Total: 24500416
Trainable: 24500416
"""