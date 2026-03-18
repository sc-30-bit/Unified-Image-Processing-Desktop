from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_DIR = ROOT_DIR / "weights"


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    default_model: str
    model_choices: tuple[str, ...]
    description: str


TASK_SPECS = {
    "superres": TaskSpec(
        key="superres",
        title="Super Resolution",
        default_model="realesrgan-x4plus.onnx",
        model_choices=("realesrgan-x4plus.onnx",),
        description="Upscale the image with tiled Real-ESRGAN inference.",
    ),
    "dehaze": TaskSpec(
        key="dehaze",
        title="Dehaze",
        default_model="dehazeformer-s-final.onnx",
        model_choices=("dehazeformer-s-final.onnx",),
        description="Remove haze with DehazeFormer.",
    ),
    "derain": TaskSpec(
        key="derain",
        title="Derain",
        default_model="restormer_rain_raw.onnx",
        model_choices=("restormer_rain_raw.onnx",),
        description="Rain removal based on Restormer.",
    ),
    "desnow": TaskSpec(
        key="desnow",
        title="Desnow",
        default_model="restormer_rain_raw.onnx",
        model_choices=("restormer_rain_raw.onnx",),
        description="Snow removal placeholder using the available Restormer weight.",
    ),
    "underwater": TaskSpec(
        key="underwater",
        title="Underwater Enhance",
        default_model="funie_gan_sim.onnx",
        model_choices=("funie_gan_sim.onnx",),
        description="Enhance underwater images with FUnIE-GAN.",
    ),
    "old_photo": TaskSpec(
        key="old_photo",
        title="Old Photo Restore",
        default_model="realesrgan-x4plus.onnx",
        model_choices=("realesrgan-x4plus.onnx",),
        description="Basic restoration and upscale path for old photos.",
    ),
    "colorization": TaskSpec(
        key="colorization",
        title="Colorization",
        default_model="deoldify_artistic_512.onnx",
        model_choices=("deoldify_artistic_512.onnx",),
        description="Colorize grayscale or old photos with DeOldify.",
    ),
    "style": TaskSpec(
        key="style",
        title="Style Transfer",
        default_model="style_candy.onnx",
        model_choices=(
            "style_candy.onnx",
            "style_mosaic.onnx",
            "style_rain_princess.onnx",
            "style_udnie.onnx",
        ),
        description="Run feed-forward style transfer.",
    ),
    "detect": TaskSpec(
        key="detect",
        title="Object Detection",
        default_model="yolov8n.onnx",
        model_choices=("yolov8n.onnx",),
        description="YOLOv8 detection with box rendering.",
    ),
    "segment": TaskSpec(
        key="segment",
        title="Instance Segmentation",
        default_model="yolov8n-seg.onnx",
        model_choices=("yolov8n-seg.onnx",),
        description="YOLOv8 segmentation with masks and boxes.",
    ),
    "track": TaskSpec(
        key="track",
        title="Object Tracking",
        default_model="yolov8n.onnx",
        model_choices=("yolov8n.onnx",),
        description="YOLOv8 detection with lightweight multi-object tracking.",
    ),
}

DEFAULT_TASK_KEY = "track"
CAMERA_TASK_KEYS = ("detect", "segment", "track")
