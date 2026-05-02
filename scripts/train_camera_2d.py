from pathlib import Path

import torch
import torchvision.ops as ops
from torch.utils.data import DataLoader

import wandb
from src.datasets.camera_2d.collate import camera_2d_collate_fn
from src.datasets.camera_2d.dataset import Camera2DDataset
from src.datasets.camera_2d.transforms import Camera2DTransform
from src.domain.enums import CameraPosition
from src.models.detector_2d import build_faster_rcnn
from src.sources.waymo.enums import WaymoCamera
from src.sources.waymo.factory import build_waymo_loaders
from src.utils.box_utils import cxcywh_to_xyxy
from src.utils.logging import get_logger, setup_logging

setup_logging("INFO")
logger = get_logger("train_camera_2d")


def log_predictions(model, batch, device, num_classes, step, score_threshold=0.5):
    """Run inference on one batch and log predictions vs ground truth to wandb."""
    model.eval()
    with torch.no_grad():
        images = [img.to(device) for img in batch["image"]]
        predictions = model(images)

    wandb_images = []
    for i, (img_tensor, pred, gt_boxes, gt_labels) in enumerate(
        zip(images, predictions, batch["boxes"], batch["labels"])
    ):
        # convert tensor back to HWC uint8 for display
        img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")

        # filter predictions by score
        keep = pred["scores"] > score_threshold
        pred_boxes_filtered = pred["boxes"][keep].cpu().numpy()
        pred_scores = pred["scores"][keep].cpu().numpy()
        pred_labels = pred["labels"][keep].cpu().numpy()

        # build wandb box annotations
        box_data = []

        # ground truth boxes (in green)
        for box, label in zip(cxcywh_to_xyxy(gt_boxes).numpy(), gt_labels.numpy()):
            box_data.append(
                {
                    "position": {
                        "minX": float(box[0]),
                        "minY": float(box[1]),
                        "maxX": float(box[2]),
                        "maxY": float(box[3]),
                    },
                    "class_id": int(label),
                    "box_caption": f"GT: {label}",
                    "scores": {"score": 1.0},
                    "domain": "pixel",
                }
            )
        # predicted boxes — already in xyxy from the model
        for box, label, score in zip(pred_boxes_filtered, pred_labels, pred_scores):
            box_data.append(
                {
                    "position": {
                        "minX": float(box[0]),
                        "minY": float(box[1]),
                        "maxX": float(box[2]),
                        "maxY": float(box[3]),
                    },
                    "class_id": int(label) + num_classes,
                    "box_caption": f"Pred: {int(label)} {score:.2f}",
                    "domain": "pixel",
                    "scores": {"score": float(score)},
                }
            )
        wandb_images.append(
            wandb.Image(
                img_np,
                boxes={"predictions": {"box_data": box_data}},
                caption=f"Sample {i} — {batch['meta'][i]['segment_id'][:20]}",
            )
        )
        wandb.log({"predictions": wandb_images}, step=step)
        model.train()


# --- config ---
MODEL = "fasterrcnn_resnet50_fpn"
DATA_ROOT = Path("/workspaces/object_detection/data/waymo/raw")
BATCH_SIZE = 2
NUM_WORKERS = 0
TARGET_SIZE = (320, 320)
NUM_EPOCHS = 2
NUM_CLASSES = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device: {DEVICE}")

# --- init wandb ---
wandb.init(
    project="waymo-2d-detection",
    config={
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "target_size": TARGET_SIZE,
        "num_classes": NUM_CLASSES,
        "learning_rate": LEARNING_RATE,
        "model": MODEL,
    },
)

CAMERAS = [
    CameraPosition.FRONT,
    # CameraPosition.FRONT_LEFT,
    # CameraPosition.FRONT_RIGHT,
    # CameraPosition.SIDE_LEFT,
    # CameraPosition.SIDE_RIGHT,
]

# --- loaders ---
logger.info("Building Waymo frame loaders...")
train_loaders = build_waymo_loaders(
    data_root=DATA_ROOT,
    split="training",
    cameras=CAMERAS,
    load_camera_labels=True,
)
logger.info(f"Built {len(train_loaders)} segment loaders for training split.")

# --- dataset ---
transform = Camera2DTransform(target_image_size=TARGET_SIZE)
logger.info("Initializing Camera2DDataset with transforms...")
train_dataset = Camera2DDataset(
    loaders=train_loaders,
    cameras=CAMERAS,
    transform=transform,
)
logger.info("Camera2DDataset initialized successfully.")
logger.info(f"Dataset size: {len(train_dataset)} samples")
logger.info(f"Sample keys: {train_dataset[0].keys()}")

# --- dataloader ---
logger.info("Creating DataLoader for training dataset...")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=camera_2d_collate_fn,
)
logger.info("DataLoader created successfully.")

# --- model ---
logger.info("Building Faster R-CNN model...")
model = build_faster_rcnn(num_classes=NUM_CLASSES)
model.to(DEVICE)
model.train()
logger.info("Model initialized and moved to device.")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# --- training loop ---
for epoch in range(NUM_EPOCHS):
    logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        logger.info(f"Processing batch {batch_idx+1}/{len(train_loader)}")
        logger.info("Converting batch data to tensors and moving to device...")
        images = [img.to(DEVICE) for img in batch["image"]]  # list of tensors
        targets = [
            {"boxes": cxcywh_to_xyxy(boxes).to(DEVICE), "labels": labels.to(DEVICE)}
            for boxes, labels in zip(batch["boxes"], batch["labels"])
        ]
        logger.info("Batch data converted to tensors and moved to device.")
        # forward pass - Faster R-CNN returns a dict of losses in training
        logger.info("Performing forward pass through the model...")
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        logger.info(
            f"Batch {batch_idx+1} Loss: {loss.item():.4f} "
            f"  {({k: f'{v.item():.4f}' for k, v in loss_dict.items()})}"
        )

        # backward pass
        logger.info("Performing backward pass and optimizer step...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info("Optimizer step completed.")

        total_loss += loss.item()

        # log to wandb every batch
        wandb.log(
            {
                "loss/total": loss.item(),
                "loss/classifier": loss_dict["loss_classifier"].item(),
                "loss/box_reg": loss_dict["loss_box_reg"].item(),
                "loss/objectness": loss_dict["loss_objectness"].item(),
                "loss/rpn_box_reg": loss_dict["loss_rpn_box_reg"].item(),
                "epoch": epoch + 1,
                "batch": batch_idx,
            },
            step=epoch * len(train_loader) + batch_idx,
        )

        if batch_idx % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{NUM_EPOCHS} "
                f"Batch {batch_idx}/{len(train_loader)} "
                f"Loss: {loss.item():.4f} "
                f"  {({k: f'{v.item():.4f}' for k, v in loss_dict.items()})}"
            )
            if batch_idx % 50 == 0:
                log_predictions(
                    model,
                    batch,
                    DEVICE,
                    NUM_CLASSES,
                    step=epoch * len(train_loader) + batch_idx,
                )

        # explicitly free memory
        del loss_dict, loss, images, targets
        torch.cuda.empty_cache()  # no-op on CPU but good habit
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
wandb.finish()
logger.info("Training completed.")
