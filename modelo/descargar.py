"""Descarga y verifica los pesos de COCO durante la construcción."""
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


if __name__ == "__main__":
    FasterRCNN_ResNet50_FPN_Weights.DEFAULT.get_state_dict(
        progress=True, check_hash=True
    )
