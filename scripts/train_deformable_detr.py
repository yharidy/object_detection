"""Train the Deformable DETR detector on transformed Waymo camera frames."""

import os

import torch
from torch.utils.data import DataLoader

import wandb
from src.datasets.camera_2d.collate import camera_2d_collate_fn
from src.datasets.camera_2d.dataset import Camera2DDataset
from src.datasets.camera_2d.transforms import Camera2DTransform
from src.domain.enums import CameraPosition
from src.models.detr.deformable_detr import DeformableDetr
from src.models.detr.detr_loss import DETRLoss
from src.sources.waymo.factory import build_waymo_loaders
from src.utils.box_utils import cxcywh_to_xyxy
from src.utils.logging import get_logger, setup_logging

setup_logging("INFO")
logger = get_logger("train_camera_2d")


def log_predictions(model, batch, device, num_classes, step, score_threshold=0.5):
    """Log thresholded query predictions and ground-truth boxes to W&B.

    The model emits normalized ``cxcywh`` boxes and logits including the
    no-object class. This helper converts boxes to pixel-space ``xyxy`` only
    for visualization and excludes the no-object class when selecting labels.

    Args:
        model: Deformable DETR model to evaluate temporarily.
        batch: Collated batch containing images, ``cxcywh`` target boxes,
            labels, and metadata.
        device: Device on which to run the model.
        num_classes: Number of foreground classes, used for W&B class IDs.
        step: W&B step at which to log the images.
        score_threshold: Minimum foreground score for a displayed prediction.
    """
    model.eval()
    with torch.no_grad():
        images = batch["image"].to(device)
        decoder_output, _ = model(images, masks=None)
        _, pred_boxes, pred_logits = decoder_output
        probabilities = pred_logits.softmax(dim=-1)
        pred_scores, pred_labels = probabilities[..., :-1].max(dim=-1)

    # ImageNet stats — must match what you used in Camera2DTransform
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    wandb_images = []
    scale = torch.tensor(
        [TARGET_SIZE[0], TARGET_SIZE[1], TARGET_SIZE[0], TARGET_SIZE[1]],
        device=pred_boxes.device,
        dtype=pred_boxes.dtype,
    )
    for i, (img_tensor, gt_boxes, gt_labels) in enumerate(
        zip(images, batch["boxes"], batch["labels"])
    ):
        pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes[i])
        pred_boxes_xyxy = pred_boxes_xyxy * scale
        # denormalize before display
        img_display = img_tensor.cpu() * std + mean
        img_display = (img_display.clamp(0, 1) * 255).byte()
        img_np = img_display.permute(1, 2, 0).numpy()  # CHW → HWC

        # filter predictions by score
        keep = pred_scores[i] > score_threshold
        pred_boxes_filtered = pred_boxes_xyxy[keep].cpu().numpy()
        image_pred_scores = pred_scores[i][keep].cpu().numpy()
        image_pred_labels = pred_labels[i][keep].cpu().numpy()

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
        for box, label, score in zip(
            pred_boxes_filtered, image_pred_labels, image_pred_scores
        ):
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
MODEL = "deformable_detr"
DATA_ROOT = os.getenv("DATA_ROOT", "/workspaces/object_detection/data/waymo/raw")
N_SEGMENTS = int(os.getenv("N_SEGMENTS", "-1"))
if N_SEGMENTS == -1:
    logger.warning(
        "N_SEGMENTS is set to -1, which means all segments will be loaded. This may lead to long loading times and high memory usage."
    )
elif N_SEGMENTS <= 0:
    raise ValueError("N_SEGMENTS must be a positive integer or -1 for all segments.")
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
    project="waymo-deformable-detr",
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
]

# --- loaders ---
logger.info("Building Waymo frame loaders...")
train_loaders = build_waymo_loaders(
    data_root=DATA_ROOT,
    split="training",
    cameras=CAMERAS,
    load_camera_labels=True,
    num_segments=N_SEGMENTS,
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
logger.info("Building Deformable DETR model...")
model = DeformableDetr(
    num_levels=4,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_classes=NUM_CLASSES,
    num_queries=100,
    num_heads=8,
    hidden_dim=256,
)
model.to(DEVICE)
model.train()
logger.info("Model initialized and moved to device.")

# Matcher and loss
detr_loss = DETRLoss(
    image_width=TARGET_SIZE[0],
    image_height=TARGET_SIZE[1],
    num_classes=NUM_CLASSES,
    class_loss_weight=1.0,
    box_loss_weight=1.0,
    giou_loss_weight=1.0,
)
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# --- training loop ---
for epoch in range(NUM_EPOCHS):
    logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        logger.info(f"Processing batch {batch_idx+1}/{len(train_loader)}")
        logger.info("Converting batch data to tensors and moving to device...")
        images = batch["image"].to(DEVICE)
        logger.info("Batch data converted to tensors and moved to device.")
        # forward pass
        logger.info("Performing forward pass through the model...")
        decoder_output, _ = model(images, masks=None)
        _, pred_boxes, pred_class_logits = decoder_output
        # cmopute loss
        gt_boxes = [boxes.to(DEVICE) for boxes in batch["boxes"]]
        gt_labels = [labels.to(DEVICE) for labels in batch["labels"]]
        loss = detr_loss(pred_class_logits, pred_boxes, gt_labels, gt_boxes)
        logger.info(f"Batch {batch_idx+1} Loss: {loss}")

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
        del loss, images
        torch.cuda.empty_cache()
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
wandb.finish()
logger.info("Training completed.")
