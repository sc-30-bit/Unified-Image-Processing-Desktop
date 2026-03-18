from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort

try:
    from .constants import COCO_CLASSES, DEFAULT_WEIGHTS_DIR, TASK_SPECS
except ImportError:
    from constants import COCO_CLASSES, DEFAULT_WEIGHTS_DIR, TASK_SPECS


@dataclass
class InferenceResult:
    image: np.ndarray | None = None
    summary: str = ""
    detections: list[dict] = field(default_factory=list)
    tracks: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class RestorationConfig:
    input_0_1: bool = True
    output_0_1: bool = True
    use_tanh_range: bool = False
    input_name: str = "input"
    output_name: str = "output"
    tile_size: int = 0
    tile_overlap: int = 32
    scale: int = 1


def available_providers() -> list[str]:
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_restoration_config(mode: str) -> RestorationConfig:
    if mode == "superres":
        return RestorationConfig(tile_size=256, tile_overlap=32, scale=4)
    if mode == "old_photo":
        return RestorationConfig(tile_size=256, tile_overlap=32, scale=4)
    if mode in {"dehaze", "underwater"}:
        return RestorationConfig(
            input_0_1=False,
            output_0_1=False,
            use_tanh_range=True,
            tile_size=0,
            tile_overlap=32,
            scale=1,
        )
    if mode == "style":
        return RestorationConfig(
            input_0_1=False,
            output_0_1=False,
            use_tanh_range=False,
            tile_size=0,
            tile_overlap=32,
            scale=1,
        )
    return RestorationConfig(tile_size=0, tile_overlap=32, scale=1)


class SessionWrapper:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=available_providers(),
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]


class RestorationModel(SessionWrapper):
    def __init__(self, model_path: Path, config: RestorationConfig):
        super().__init__(model_path)
        self.config = config
        self.input_name = config.input_name if config.input_name in self.input_names else self.input_names[0]
        self.output_name = config.output_name if config.output_name in self.output_names else self.output_names[0]

    def predict(self, image: np.ndarray) -> InferenceResult:
        if image is None or image.size == 0:
            return InferenceResult(summary="Empty input image.")

        h, w = image.shape[:2]
        tile = self.config.tile_size
        overlap = self.config.tile_overlap

        if tile <= 0 or (h <= tile and w <= tile):
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32
            inputs = image
            if pad_h > 0 or pad_w > 0:
                inputs = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            pred = self._infer_single(inputs)
            if pred is None:
                return InferenceResult(summary="Model inference failed.")
            if pad_h > 0 or pad_w > 0:
                scale = pred.shape[0] // inputs.shape[0]
                pred = pred[: h * scale, : w * scale].copy()
            return InferenceResult(image=pred, summary=f"Processed {w}x{h} image.")

        h_indices = self._generate_indices(h, tile, overlap)
        w_indices = self._generate_indices(w, tile, overlap)
        output_acc = None
        count_map = None
        scale = 1

        for h_idx in h_indices:
            for w_idx in w_indices:
                h_end = min(h, h_idx + tile)
                w_end = min(w, w_idx + tile)
                patch = image[h_idx:h_end, w_idx:w_end].copy()
                out_patch = self._infer_single(patch)
                if out_patch is None:
                    continue

                if output_acc is None:
                    scale = out_patch.shape[0] // patch.shape[0]
                    output_acc = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
                    count_map = np.zeros_like(output_acc, dtype=np.float32)

                out_patch_f = out_patch.astype(np.float32)
                y0 = h_idx * scale
                x0 = w_idx * scale
                y1 = y0 + out_patch.shape[0]
                x1 = x0 + out_patch.shape[1]
                output_acc[y0:y1, x0:x1] += out_patch_f
                count_map[y0:y1, x0:x1] += 1.0

        if output_acc is None or count_map is None:
            return InferenceResult(summary="Model inference failed.")

        merged = np.divide(output_acc, np.maximum(count_map, 1e-6)).clip(0, 255).astype(np.uint8)
        return InferenceResult(image=merged, summary=f"Tiled inference finished at x{scale}.")

    @staticmethod
    def _generate_indices(length: int, tile_size: int, overlap: int) -> list[int]:
        stride = max(tile_size - overlap, 1)
        indices = list(range(0, max(length - tile_size, 0), stride))
        indices.append(max(length - tile_size, 0))
        deduped: list[int] = []
        for item in indices:
            if not deduped or deduped[-1] != item:
                deduped.append(item)
        return deduped

    def _infer_single(self, image: np.ndarray) -> np.ndarray | None:
        scale_factor = 1.0
        mean = np.zeros(3, dtype=np.float32)

        if self.config.use_tanh_range:
            scale_factor = 1.0 / 127.5
            mean[:] = 127.5
        elif self.config.input_0_1:
            scale_factor = 1.0 / 255.0

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - mean) * scale_factor
        blob = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        out = outputs[0]
        if out.ndim != 4 or out.shape[1] != 3:
            return None

        data = out[0]
        rgb_out = np.transpose(data, (1, 2, 0)).astype(np.float32)

        range_is_01 = False
        if not self.config.use_tanh_range:
            center = float(rgb_out[rgb_out.shape[0] // 2, rgb_out.shape[1] // 2, 0])
            range_is_01 = -2.0 < center < 2.0

        if self.config.use_tanh_range:
            rgb_out = (rgb_out * 0.5 + 0.5) * 255.0
        elif self.config.output_0_1 or range_is_01:
            rgb_out = rgb_out * 255.0

        rgb_out = np.clip(rgb_out, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)


class ColorizationModel(SessionWrapper):
    def predict(self, image: np.ndarray) -> InferenceResult:
        if image is None or image.size == 0:
            return InferenceResult(summary="Empty input image.")

        input_shape = self.session.get_inputs()[0].shape
        net_h = int(input_shape[2]) if len(input_shape) >= 4 and isinstance(input_shape[2], int) and input_shape[2] > 0 else 512
        net_w = int(input_shape[3]) if len(input_shape) >= 4 and isinstance(input_shape[3], int) and input_shape[3] > 0 else 512

        orig_h, orig_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        resized = cv2.resize(gray_rgb, (net_w, net_h)).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (resized - mean) / std
        blob = np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)

        out = self.session.run(self.output_names, {self.input_names[0]: blob})[0]
        rgb = self._parse_color_output(out, std, mean)
        if rgb is None:
            return InferenceResult(summary="Unexpected colorization output shape.")

        color_rgb = cv2.resize(rgb, (orig_w, orig_h))
        orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color_yuv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2YUV)
        orig_yuv = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2YUV)
        fused = orig_yuv.copy()
        fused[..., 1:] = color_yuv[..., 1:]
        final_rgb = cv2.cvtColor(fused, cv2.COLOR_YUV2RGB)
        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
        return InferenceResult(image=final_bgr, summary="Colorization finished.")

    @staticmethod
    def _parse_color_output(out: np.ndarray, std: np.ndarray, mean: np.ndarray) -> np.ndarray | None:
        if out.ndim != 4:
            return None

        if out.shape[1] == 3:
            rgb = np.transpose(out[0], (1, 2, 0))
        elif out.shape[-1] == 3:
            rgb = out[0]
        else:
            return None

        rgb = rgb * std + mean
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)


class YoloModel(SessionWrapper):
    def __init__(self, model_path: Path, labels: Iterable[str] | None = None):
        super().__init__(model_path)
        self.labels = list(labels or COCO_CLASSES)
        self.input_size = (640, 640)
        self.mask_threshold = 0.5
        self.is_segmentation = len(self.output_names) >= 2

    def detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45, max_det: int = 100) -> list[dict]:
        preprocessed, scale = self._preprocess(image)
        blob = np.transpose(preprocessed.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        outputs = self.session.run(self.output_names, {self.input_names[0]: blob})
        detections = self._parse_detections(outputs[0], image.shape, scale, conf)
        kept = self._nms(detections, conf, iou, max_det)
        proto = outputs[1][0] if self.is_segmentation and len(outputs) >= 2 else None
        final_dets: list[dict] = []

        for idx in kept:
            det = detections[idx]
            det = dict(det)
            if proto is not None and det["mask_coefs"] is not None:
                det["mask"] = self._process_mask(proto, det["mask_coefs"], det["box"], scale)
            else:
                det["mask"] = None
            final_dets.append(det)
        return final_dets

    def render(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        annotated = image.copy()
        for det in detections:
            if det.get("mask") is not None:
                annotated = self._overlay_mask(annotated, det["mask"], det["box"], self._color_for_class(det["class_id"]))
            self._draw_box(annotated, det)
        return annotated

    def predict(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45, max_det: int = 100) -> InferenceResult:
        if image is None or image.size == 0:
            return InferenceResult(summary="Empty input image.")
        detections = self.detect(image, conf=conf, iou=iou, max_det=max_det)
        annotated = self.render(image, detections)
        return InferenceResult(image=annotated, summary=f"{len(detections)} object(s) detected.", detections=detections)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        ih, iw = rgb.shape[:2]
        scale = max(iw / self.input_size[0], ih / self.input_size[1])
        new_w = int(iw / scale)
        new_h = int(ih / scale)
        resized = cv2.resize(rgb, (new_w, new_h))
        canvas = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        return canvas, scale

    def _parse_detections(self, output0: np.ndarray, image_shape: tuple[int, ...], scale: float, conf: float) -> list[dict]:
        dims = output0.shape[1]
        rows = output0.shape[2]
        transposed = output0[0].T
        detections: list[dict] = []
        image_h, image_w = image_shape[:2]

        for row in range(rows):
            pdata = transposed[row]
            scores = pdata[4 : 4 + len(self.labels)]
            cls = int(np.argmax(scores))
            score = float(scores[cls])
            if score < conf:
                continue

            x, y, w, h = map(float, pdata[:4])
            left = max(0, int((x - 0.5 * w) * scale))
            top = max(0, int((y - 0.5 * h) * scale))
            width = min(image_w - left, int(w * scale))
            height = min(image_h - top, int(h * scale))
            if width <= 1 or height <= 1:
                continue

            mask_coefs = None
            if self.is_segmentation and dims >= 32:
                mask_coefs = pdata[dims - 32 : dims].astype(np.float32).copy()

            detections.append(
                {
                    "class_id": cls,
                    "label": self.labels[cls] if cls < len(self.labels) else f"class_{cls}",
                    "confidence": score,
                    "box": (left, top, width, height),
                    "mask_coefs": mask_coefs,
                }
            )
        return detections

    @staticmethod
    def _nms(detections: list[dict], conf_threshold: float, iou_threshold: float, max_det: int) -> list[int]:
        if not detections:
            return []
        boxes = [det["box"] for det in detections]
        scores = [det["confidence"] for det in detections]
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        if len(indices) == 0:
            return []
        flat = [int(idx[0] if isinstance(idx, (tuple, list, np.ndarray)) else idx) for idx in indices]
        return flat[:max_det]

    def _process_mask(self, protos: np.ndarray, mask_coefs: np.ndarray, box: tuple[int, int, int, int], scale: float) -> np.ndarray | None:
        left, top, width, height = box
        if width <= 0 or height <= 0:
            return None
        mask_mat = mask_coefs @ protos.reshape(32, -1)
        mask_mat = sigmoid(mask_mat).reshape(160, 160)

        x = int(left / scale * 0.25)
        y = int(top / scale * 0.25)
        w = int(width / scale * 0.25)
        h = int(height / scale * 0.25)
        x = max(0, x)
        y = max(0, y)
        w = min(160 - x, w)
        h = min(160 - y, h)
        if w <= 0 or h <= 0:
            return None

        cropped = mask_mat[y : y + h, x : x + w]
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized > self.mask_threshold

    @staticmethod
    def _overlay_mask(image: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> np.ndarray:
        left, top, width, height = box
        overlay = image.copy()
        roi = overlay[top : top + height, left : left + width]
        roi[mask] = (roi[mask] * 0.45 + np.array(color, dtype=np.float32) * 0.55).astype(np.uint8)
        overlay[top : top + height, left : left + width] = roi
        return overlay

    @staticmethod
    def _draw_box(image: np.ndarray, det: dict) -> None:
        left, top, width, height = det["box"]
        track_id = det.get("track_id")
        label = det["label"]
        if track_id is not None:
            label = f"ID {track_id} | {label}"
        color = YoloModel._color_for_class(track_id if track_id is not None else det["class_id"])
        cv2.rectangle(image, (left, top), (left + width, top + height), color, 2)
        text = f'{label} {det["confidence"]:.2f}'
        font_scale = 0.55
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_h = th + baseline + 8
        label_w = min(width, tw + 8)
        if label_w < 40:
            label_w = min(tw + 8, image.shape[1] - left)
        label_y1 = top + min(label_h, height)
        cv2.rectangle(image, (left, top), (left + label_w, label_y1), color, -1)
        cv2.putText(
            image,
            text,
            (left + 4, min(top + th + 3, top + height - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (15, 18, 25),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _color_for_class(class_id: int) -> tuple[int, int, int]:
        palette = [
            (41, 128, 185),
            (22, 160, 133),
            (211, 84, 0),
            (192, 57, 43),
            (142, 68, 173),
            (39, 174, 96),
        ]
        return palette[class_id % len(palette)]


@dataclass
class TrackState:
    track_id: int
    class_id: int
    kf: cv2.KalmanFilter
    box: tuple[int, int, int, int]
    center: tuple[float, float]
    velocity: tuple[float, float]
    missed: int = 0
    hits: int = 1
    history: list[tuple[float, float]] = field(default_factory=list)


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.15, max_missed: int = 20):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, TrackState] = {}

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()

    def update(self, detections: list[dict]) -> list[dict]:
        if not self.tracks:
            for det in detections:
                self._start_track(det)
            return detections

        track_ids = list(self.tracks.keys())
        predicted_boxes: dict[int, tuple[int, int, int, int]] = {}
        cost_matrix = np.full((len(track_ids), len(detections)), 1e6, dtype=np.float32)

        for row, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            predicted_box = self._predict_track(track)
            predicted_boxes[track_id] = predicted_box
            predicted_center = _box_center(predicted_box)
            for col, det in enumerate(detections):
                if det["class_id"] != track.class_id:
                    continue
                iou = _box_iou(predicted_box, det["box"])
                if iou < self.iou_threshold:
                    continue
                det_center = _box_center(det["box"])
                motion_penalty = _center_distance(predicted_center, det_center) / max(max(predicted_box[2], predicted_box[3]), 1.0)
                cost_matrix[row, col] = (1.0 - iou) + 0.05 * motion_penalty

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        if len(track_ids) > 0 and len(detections) > 0:
            assignments = _hungarian(cost_matrix)
            for row, col in assignments:
                if cost_matrix[row, col] >= 1e5:
                    continue
                track_id = track_ids[row]
                track = self.tracks[track_id]
                det = detections[col]
                prev_center = track.center
                corrected_box = self._correct_track(track, det["box"])
                cx, cy = _box_center(corrected_box)
                track.velocity = (cx - prev_center[0], cy - prev_center[1])
                track.center = (cx, cy)
                track.box = corrected_box
                track.missed = 0
                track.hits += 1
                track.history.append((cx, cy))
                track.history = track.history[-30:]
                det["track_id"] = track_id
                det["trail"] = list(track.history)
                det["box"] = corrected_box
                matched_tracks.add(track_id)
                matched_dets.add(col)

        for track_id in list(self.tracks.keys()):
            if track_id in matched_tracks:
                continue
            track = self.tracks[track_id]
            track.box = predicted_boxes.get(track_id, track.box)
            track.center = _box_center(track.box)
            track.missed += 1
            if track.missed > self.max_missed:
                self.tracks.pop(track_id, None)

        for idx, det in enumerate(detections):
            if idx in matched_dets:
                continue
            self._start_track(det)

        return detections

    def _start_track(self, det: dict) -> None:
        cx, cy = _box_center(det["box"])
        track_id = self.next_id
        self.next_id += 1
        kf = _create_kalman_filter(cx, cy, det["box"][2], det["box"][3])
        history = [(cx, cy)]
        self.tracks[track_id] = TrackState(
            track_id=track_id,
            class_id=det["class_id"],
            kf=kf,
            box=det["box"],
            center=(cx, cy),
            velocity=(0.0, 0.0),
            missed=0,
            hits=1,
            history=history,
        )
        det["track_id"] = track_id
        det["trail"] = history

    @staticmethod
    def _predict_track(track: TrackState) -> tuple[int, int, int, int]:
        prediction = track.kf.predict()
        cx = float(prediction[0, 0])
        cy = float(prediction[1, 0])
        w = max(2.0, float(prediction[2, 0]))
        h = max(2.0, float(prediction[3, 0]))
        return _box_from_center(cx, cy, w, h)

    @staticmethod
    def _correct_track(track: TrackState, det_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        cx, cy = _box_center(det_box)
        measurement = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(det_box[2])], [np.float32(det_box[3])]])
        corrected = track.kf.correct(measurement)
        return _box_from_center(
            float(corrected[0, 0]),
            float(corrected[1, 0]),
            max(2.0, float(corrected[2, 0])),
            max(2.0, float(corrected[3, 0])),
        )


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    left, top, width, height = box
    return left + width * 0.5, top + height * 0.5


def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _box_from_center(cx: float, cy: float, width: float, height: float) -> tuple[int, int, int, int]:
    return (
        int(cx - width * 0.5),
        int(cy - height * 0.5),
        int(width),
        int(height),
    )


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = aw * ah + bw * bh - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _create_kalman_filter(cx: float, cy: float, width: float, height: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(8, 4)
    kf.transitionMatrix = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(8, dtype=np.float32)
    kf.statePost = np.array([[cx], [cy], [width], [height], [0], [0], [0], [0]], dtype=np.float32)
    return kf


def _hungarian(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    if cost_matrix.size == 0:
        return []
    rows, cols = cost_matrix.shape
    transposed = False
    cost = cost_matrix.copy()
    if rows > cols:
        cost = cost.T
        rows, cols = cost.shape
        transposed = True

    u = np.zeros(rows + 1, dtype=np.float32)
    v = np.zeros(cols + 1, dtype=np.float32)
    p = np.zeros(cols + 1, dtype=np.int32)
    way = np.zeros(cols + 1, dtype=np.int32)

    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(cols + 1, np.inf, dtype=np.float32)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignments: list[tuple[int, int]] = []
    for j in range(1, cols + 1):
        if p[j] == 0:
            continue
        row = p[j] - 1
        col = j - 1
        if transposed:
            assignments.append((col, row))
        else:
            assignments.append((row, col))
    return assignments


class InferenceEngine:
    def __init__(self, weights_dir: Path | None = None):
        self.weights_dir = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        self._cache: dict[tuple[str, str], object] = {}
        self._tracker = SimpleTracker()

    def model_choices(self, task_key: str) -> tuple[str, ...]:
        return TASK_SPECS[task_key].model_choices

    def reset_tracker(self) -> None:
        self._tracker.reset()

    def run(self, task_key: str, model_name: str, image: np.ndarray) -> InferenceResult:
        model = self._get_model(task_key, model_name)
        if task_key == "track":
            detections = model.detect(image)
            tracked = self._tracker.update(detections)
            annotated = model.render(image, tracked)
            self._draw_trails(annotated, tracked)
            tracks = [{"track_id": det["track_id"], "label": det["label"], "box": det["box"]} for det in tracked]
            return InferenceResult(image=annotated, summary=f"{len(tracks)} tracked object(s).", detections=tracked, tracks=tracks)
        return model.predict(image)

    def _get_model(self, task_key: str, model_name: str):
        resolved_task = "detect" if task_key == "track" else task_key
        cache_key = (resolved_task, model_name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        model_path = self.weights_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if resolved_task in {"detect", "segment"}:
            model = YoloModel(model_path)
        elif resolved_task == "colorization":
            model = ColorizationModel(model_path)
        else:
            model = RestorationModel(model_path, make_restoration_config(resolved_task))

        self._cache[cache_key] = model
        return model

    @staticmethod
    def _draw_trails(image: np.ndarray, detections: list[dict]) -> None:
        for det in detections:
            trail = det.get("trail") or []
            if len(trail) < 2:
                continue
            color = YoloModel._color_for_class(det["track_id"])
            for i in range(1, len(trail)):
                p0 = tuple(int(v) for v in trail[i - 1])
                p1 = tuple(int(v) for v in trail[i])
                cv2.line(image, p0, p1, color, 2, cv2.LINE_AA)
