from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .constants import CAMERA_TASK_KEYS, DEFAULT_TASK_KEY, DEFAULT_WEIGHTS_DIR, TASK_SPECS
from .inference import InferenceEngine, InferenceResult


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
SOURCE_MODES = ("image", "video", "camera")


def read_image_file(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def first_video_frame(path: Path) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return None
    return frame


def qimage_from_bgr(image: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    return QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()


class WorkerSignals(QObject):
    status_changed = pyqtSignal(str, str)
    progress_changed = pyqtSignal(float, str, str)
    result_ready = pyqtSignal(object)
    preview_ready = pyqtSignal(object, object)
    live_meta = pyqtSignal(str, str)
    camera_bootstrap = pyqtSignal(object)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)


class DropSurface(QFrame):
    files_dropped = pyqtSignal(list)
    clicked = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropSurface")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 26)
        layout.setSpacing(8)

        badge = QLabel("DRAG AND DROP")
        badge.setObjectName("dropBadge")
        badge.setAlignment(Qt.AlignCenter)

        title = QLabel("Drop images or a video here")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Drop a single image or a single video.")
        subtitle.setObjectName("dropSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignCenter)

        action = QLabel("You can also click to browse files.")
        action.setObjectName("dropHint")
        action.setAlignment(Qt.AlignCenter)

        layout.addStretch(1)
        layout.addWidget(badge)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(action)
        layout.addStretch(1)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("dragging", True)
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self.setProperty("dragging", False)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        self.setProperty("dragging", False)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(Path(url.toLocalFile()))
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
            return
        event.ignore()


class PreviewCard(QFrame):
    def __init__(self, title: str, subtitle: str) -> None:
        super().__init__()
        self._qimage: QImage | None = None

        self.setObjectName("previewCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        top = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("previewTitle")
        meta_label = QLabel(subtitle)
        meta_label.setObjectName("previewMeta")
        top.addWidget(title_label)
        top.addStretch(1)
        top.addWidget(meta_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setObjectName("previewImage")
        self.image_label.setMinimumSize(420, 300)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addLayout(top)
        layout.addWidget(self.image_label, 1)
        self.set_placeholder(f"{title}\nPreview")

    def set_placeholder(self, text: str) -> None:
        self._qimage = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(text)

    def set_image(self, image: np.ndarray) -> None:
        self._qimage = qimage_from_bgr(image)
        self._redraw()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._redraw()

    def _redraw(self) -> None:
        if self._qimage is None:
            return
        pixmap = QPixmap.fromImage(self._qimage).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setText("")
        self.image_label.setPixmap(pixmap)


class VisionDesktopApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.engine = InferenceEngine(DEFAULT_WEIGHTS_DIR)
        self.current_input_path: Path | None = None
        self.current_input_image: np.ndarray | None = None
        self.current_result_image: np.ndarray | None = None
        self.selected_paths: list[Path] = []
        self.output_dir: Path | None = None
        self._all_task_keys = list(TASK_SPECS.keys())
        self.last_result_video_path: Path | None = None
        self.camera_capture: cv2.VideoCapture | None = None
        self.camera_running = False
        self.stop_requested = False
        self.worker_running = False
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_video_playback)
        self.original_playback_capture: cv2.VideoCapture | None = None
        self.result_playback_capture: cv2.VideoCapture | None = None

        self.setWindowTitle("AI Vision Studio")
        self.resize(1560, 980)
        self.setMinimumSize(1240, 820)

        self._build_ui()
        self._apply_styles()
        self._refresh_task_details()
        self._refresh_source_ui()
        self._sync_action_state()

    def _build_ui(self) -> None:
        page = QWidget()
        self.setCentralWidget(page)

        root = QHBoxLayout(page)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setObjectName("sidebarScroll")
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setMinimumWidth(420)
        sidebar_scroll.setMaximumWidth(460)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(18, 18, 18, 18)
        side_layout.setSpacing(14)
        sidebar_scroll.setWidget(sidebar)

        brand = QLabel("AI Vision Studio")
        brand.setObjectName("brandTitle")
        brand_caption = QLabel("Local ONNX image and video workflow.")
        brand_caption.setObjectName("brandCaption")
        brand_caption.setWordWrap(True)
        side_layout.addWidget(brand)
        side_layout.addWidget(brand_caption)

        self.task_combo = QComboBox()
        self.task_combo.addItems(self._all_task_keys)
        self.task_combo.setCurrentText(DEFAULT_TASK_KEY)
        self.model_combo = QComboBox()
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(SOURCE_MODES))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 16)
        self.weights_edit = QLineEdit(str(DEFAULT_WEIGHTS_DIR))
        self.output_edit = QLineEdit()

        side_layout.addWidget(self._panel_title("Workflow"))
        side_layout.addWidget(self._field_block("Task", self.task_combo))
        side_layout.addWidget(self._field_block("Model", self.model_combo))
        side_layout.addWidget(self._field_block("Source Mode", self.source_combo))
        self.camera_block = self._field_block("Camera Device", self.camera_spin)
        side_layout.addWidget(self.camera_block)

        side_layout.addWidget(self._panel_title("Paths"))
        self.weights_block = self._field_block("Weights Folder", self.weights_edit)
        side_layout.addWidget(self.weights_block)

        self.path_buttons_layout = QHBoxLayout()
        self.path_buttons_layout.setSpacing(10)
        self.weights_button = QPushButton("Browse Weights")
        self.output_button = QPushButton("Output Folder")
        self.path_buttons_layout.addWidget(self.weights_button)
        self.path_buttons_layout.addWidget(self.output_button)
        side_layout.addLayout(self.path_buttons_layout)

        self.output_block = self._field_block("Output Folder", self.output_edit)
        side_layout.addWidget(self.output_block)

        side_layout.addWidget(self._panel_title("Selection"))
        self.source_summary = QLabel("No source selected")
        self.source_summary.setWordWrap(True)
        self.source_summary.setObjectName("mutedText")
        side_layout.addWidget(self.source_summary)

        self.choose_button = QPushButton("Choose Files")
        self.run_button = QPushButton("Run")
        self.stop_button = QPushButton("Stop")
        self.camera_button = QPushButton("Start Camera")
        self.save_button = QPushButton("Save Result")
        self.run_button.setObjectName("primaryButton")

        action_grid = QGridLayout()
        action_grid.setHorizontalSpacing(10)
        action_grid.setVerticalSpacing(10)
        action_grid.addWidget(self.choose_button, 0, 0, 1, 2)
        action_grid.addWidget(self.run_button, 1, 0, 1, 2)
        action_grid.addWidget(self.stop_button, 2, 0)
        action_grid.addWidget(self.camera_button, 2, 1)
        action_grid.addWidget(self.save_button, 3, 0, 1, 2)
        side_layout.addLayout(action_grid)
        side_layout.addStretch(1)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        hero = QFrame()
        hero.setObjectName("hero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 20, 24, 20)

        hero_copy = QVBoxLayout()
        hero_copy.setSpacing(6)
        hero_title = QLabel("Drag in media, switch tasks, run locally")
        hero_title.setObjectName("heroTitle")
        hero_subtitle = QLabel("Cleaner workflow, larger previews, and less sidebar clutter.")
        hero_subtitle.setObjectName("heroSubtitle")
        hero_subtitle.setWordWrap(True)
        hero_copy.addWidget(hero_title)
        hero_copy.addWidget(hero_subtitle)

        badges = QVBoxLayout()
        badges.setSpacing(10)
        badges.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.task_badge = QLabel()
        self.task_badge.setObjectName("heroBadge")
        self.mode_badge = QLabel()
        self.mode_badge.setObjectName("heroBadgeMuted")
        badges.addWidget(self.task_badge, 0, Qt.AlignRight)
        badges.addWidget(self.mode_badge, 0, Qt.AlignRight)

        hero_layout.addLayout(hero_copy, 1)
        hero_layout.addLayout(badges)

        self.drop_surface = DropSurface()

        status_bar = QFrame()
        status_bar.setObjectName("statusStrip")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(18, 14, 18, 14)
        status_layout.setSpacing(16)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusTitle")
        self.info_label = QLabel("Choose media or drag it onto the drop zone.")
        self.info_label.setObjectName("mutedText")
        self.info_label.setWordWrap(True)
        self.progress_label = QLabel("Idle")
        self.progress_label.setObjectName("chip")
        self.fps_label = QLabel("FPS: -")
        self.fps_label.setObjectName("chip")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(180)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.info_label, 1)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.progress_label)
        status_layout.addWidget(self.fps_label)

        preview_grid = QGridLayout()
        preview_grid.setContentsMargins(0, 0, 0, 0)
        preview_grid.setHorizontalSpacing(16)
        self.original_card = PreviewCard("Original", "Source preview")
        self.result_card = PreviewCard("Result", "Inference preview")
        preview_grid.addWidget(self.original_card, 0, 0)
        preview_grid.addWidget(self.result_card, 0, 1)
        preview_grid.setColumnStretch(0, 1)
        preview_grid.setColumnStretch(1, 1)

        content_layout.addWidget(hero)
        content_layout.addWidget(self.drop_surface)
        content_layout.addWidget(status_bar)
        content_layout.addLayout(preview_grid, 1)

        root.addWidget(sidebar_scroll)
        root.addWidget(content, 1)

        self.task_combo.currentTextChanged.connect(self._refresh_task_details)
        self.source_combo.currentTextChanged.connect(self._refresh_source_ui)
        self.weights_button.clicked.connect(self.choose_weights_dir)
        self.output_button.clicked.connect(self.choose_output_dir)
        self.choose_button.clicked.connect(self.choose_source)
        self.run_button.clicked.connect(self.run_current_mode)
        self.stop_button.clicked.connect(self.request_stop)
        self.camera_button.clicked.connect(self.toggle_camera)
        self.save_button.clicked.connect(self.save_result)
        self.drop_surface.clicked.connect(self.choose_source)
        self.drop_surface.files_dropped.connect(self._handle_dropped_paths)

    def _apply_styles(self) -> None:
        self.setFont(QFont("Segoe UI", 9))
        self.setStyleSheet(
            """
            QWidget {
                background: #edf2f7;
                color: #132238;
            }
            QMainWindow {
                background: #edf2f7;
            }
            QScrollArea#sidebarScroll {
                background: transparent;
            }
            QFrame#sidebar {
                background: #f7fbff;
                border: 1px solid #d9e4f0;
                border-radius: 24px;
            }
            QLabel#brandTitle {
                font-size: 20px;
                font-weight: 700;
                color: #10233e;
            }
            QLabel#brandCaption, QLabel#mutedText {
                color: #66768d;
                font-size: 10px;
            }
            QLabel#panelTitle {
                margin-top: 4px;
                font-size: 10px;
                font-weight: 700;
                color: #48607b;
                letter-spacing: 0.08em;
            }
            QFrame#fieldBlock {
                background: #ffffff;
                border: 1px solid #dce6f2;
                border-radius: 16px;
            }
            QLabel#fieldLabel {
                font-size: 10px;
                font-weight: 700;
                color: #51647d;
            }
            QComboBox, QLineEdit, QSpinBox {
                min-height: 44px;
                padding: 0 12px;
                border: 1px solid #cfdaea;
                border-radius: 12px;
                background: #ffffff;
                font-size: 10px;
            }
            QPushButton {
                min-height: 44px;
                padding: 0 14px;
                border-radius: 14px;
                border: 1px solid #cfdaea;
                background: #ffffff;
                font-weight: 600;
                font-size: 10px;
                text-align: center;
            }
            QPushButton:hover {
                background: #f3f8ff;
            }
            QPushButton#primaryButton {
                background: #1f6feb;
                color: #ffffff;
                border: none;
            }
            QPushButton#primaryButton:hover {
                background: #195dcc;
            }
            QFrame#hero {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #10243d, stop:1 #1d4f7a);
                border-radius: 28px;
            }
            QLabel#heroTitle {
                background: transparent;
                color: #f4f8ff;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#heroSubtitle {
                background: transparent;
                color: #bdd0e5;
                font-size: 12px;
            }
            QLabel#heroBadge, QLabel#heroBadgeMuted {
                padding: 8px 14px;
                border-radius: 14px;
                font-weight: 700;
                font-size: 11px;
            }
            QLabel#heroBadge {
                background: #f59e0b;
                color: #152030;
            }
            QLabel#heroBadgeMuted {
                background: rgba(255, 255, 255, 0.12);
                color: #e6eef8;
            }
            QFrame#dropSurface {
                background: #fffaf1;
                border: 2px dashed #f0b561;
                border-radius: 24px;
            }
            QFrame#dropSurface[dragging="true"] {
                background: #fff1d6;
                border: 2px dashed #d97706;
            }
            QLabel#dropBadge {
                color: #9a5d07;
                font-weight: 700;
                font-size: 11px;
                letter-spacing: 0.14em;
            }
            QLabel#dropTitle {
                color: #152030;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#dropSubtitle, QLabel#dropHint {
                color: #6f5a38;
                font-size: 12px;
            }
            QFrame#statusStrip, QFrame#previewCard {
                background: #f7fbff;
                border: 1px solid #d9e4f0;
                border-radius: 24px;
            }
            QLabel#statusTitle {
                font-size: 14px;
                font-weight: 700;
                color: #10233e;
            }
            QLabel#chip {
                padding: 6px 12px;
                border-radius: 12px;
                background: #e8f0fa;
                color: #31506f;
                font-weight: 700;
                font-size: 10px;
            }
            QLabel#previewTitle {
                font-size: 14px;
                font-weight: 700;
                color: #10233e;
            }
            QLabel#previewMeta {
                color: #6a7d95;
                font-size: 10px;
                font-weight: 600;
            }
            QLabel#previewImage {
                background: #101c2d;
                border-radius: 18px;
                color: #70839d;
                font-size: 16px;
                font-weight: 600;
            }
            QProgressBar {
                height: 10px;
                border-radius: 5px;
                background: #dce7f2;
            }
            QProgressBar::chunk {
                border-radius: 5px;
                background: #1f6feb;
            }
            """
        )

    def _panel_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("panelTitle")
        return label

    def _field_block(self, label_text: str, widget: QWidget) -> QFrame:
        frame = QFrame()
        frame.setObjectName("fieldBlock")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)
        label = QLabel(label_text)
        label.setObjectName("fieldLabel")
        layout.addWidget(label)
        layout.addWidget(widget)
        return frame

    def _refresh_task_details(self) -> None:
        self._stop_video_playback()
        task_key = self.task_combo.currentText()
        if not task_key:
            return
        spec = TASK_SPECS[task_key]
        self.task_badge.setText(spec.title)
        self.mode_badge.setText(f"Source {self.source_combo.currentText().title()}")
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(list(spec.model_choices))
        self.model_combo.setCurrentText(spec.default_model)
        self.model_combo.blockSignals(False)
        self.info_label.setText(spec.description)

    def _refresh_source_ui(self) -> None:
        self._stop_video_playback()
        mode = self.source_combo.currentText()
        needs_output = mode == "video"
        allowed_task_keys = list(CAMERA_TASK_KEYS if mode == "camera" else self._all_task_keys)
        current_task = self.task_combo.currentText()
        if [self.task_combo.itemText(i) for i in range(self.task_combo.count())] != allowed_task_keys:
            self.task_combo.blockSignals(True)
            self.task_combo.clear()
            self.task_combo.addItems(allowed_task_keys)
            self.task_combo.blockSignals(False)
        if current_task not in allowed_task_keys:
            fallback_task = DEFAULT_TASK_KEY if DEFAULT_TASK_KEY in allowed_task_keys else allowed_task_keys[0]
            self.task_combo.setCurrentText(fallback_task)
            current_task = fallback_task
        elif self.task_combo.currentText() != current_task:
            self.task_combo.setCurrentText(current_task)
        self.mode_badge.setText(f"Source {mode.title()}")
        self.camera_spin.setEnabled(mode == "camera")
        self.camera_block.setVisible(mode == "camera")
        self.camera_button.setEnabled(mode == "camera")
        self.choose_button.setEnabled(mode != "camera")
        self.output_block.setVisible(needs_output)
        self.output_button.setVisible(needs_output)
        if mode == "camera":
            self.source_summary.setText(f"Live camera will open from device {self.camera_spin.value()}.")
        elif not self.selected_paths:
            self.source_summary.setText("No source selected")
        self._sync_action_state()

    def _sync_action_state(self) -> None:
        has_result = self.current_result_image is not None
        camera_mode = self.source_combo.currentText() == "camera"
        self.run_button.setEnabled((not self.worker_running) and (camera_mode or bool(self.selected_paths)))
        self.stop_button.setEnabled(self.worker_running or self.camera_running)
        self.camera_button.setText("Stop Camera" if self.camera_running else "Start Camera")
        self.save_button.setEnabled(has_result)
        self.save_button.setText("Save Video" if self.source_combo.currentText() == "video" else "Save Result")

    def choose_source(self) -> None:
        self._stop_video_playback()
        mode = self.source_combo.currentText()
        if mode == "image":
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Choose image",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)",
            )
        elif mode == "video":
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Choose video",
                "",
                "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;All files (*.*)",
            )
        else:
            return
        if not paths:
            return
        self._ingest_paths([Path(item) for item in paths])

    def _handle_dropped_paths(self, paths: list[Path]) -> None:
        files = [path for path in paths if path.is_file()]
        if not files:
            return
        suffixes = {path.suffix.lower() for path in files}
        if len(files) == 1 and suffixes <= IMAGE_EXTS:
            self.source_combo.setCurrentText("image")
        elif len(files) == 1 and suffixes <= VIDEO_EXTS:
            self.source_combo.setCurrentText("video")
        else:
            QMessageBox.warning(self, "Unsupported drop", "Drop a single image or a single video file.")
            return
        self._ingest_paths(files)

    def _ingest_paths(self, paths: list[Path]) -> None:
        self._stop_video_playback()
        self.last_result_video_path = None
        mode = self.source_combo.currentText()
        if mode == "image":
            path = paths[0]
            image = read_image_file(path)
            if image is None:
                self._show_error("The selected image could not be decoded.")
                return
            self.selected_paths = [path]
            self.current_input_path = path
            self.current_input_image = image
            self.current_result_image = None
            self.source_summary.setText(f"{path.name}\n{image.shape[1]} x {image.shape[0]}")
            self.status_label.setText("Image loaded")
            self.info_label.setText("Ready to process the selected image.")
            self.original_card.set_image(image)
            self.result_card.set_placeholder("Result\nPreview")
        elif mode == "video":
            path = paths[0]
            frame = first_video_frame(path)
            if frame is None:
                self._show_error("Could not read the selected video.")
                return
            self.selected_paths = [path]
            self.current_input_path = path
            self.current_input_image = frame
            self.current_result_image = None
            self.source_summary.setText(f"{path.name}\nVideo source selected")
            self.status_label.setText("Video loaded")
            self.info_label.setText("Ready to process the selected video.")
            self.original_card.set_image(frame)
            self.result_card.set_placeholder("Result\nPreview")
        self._set_progress(0.0, "Idle", "FPS: -")
        self._sync_action_state()

    def choose_weights_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose weights folder", self.weights_edit.text())
        if not path:
            return
        self.weights_edit.setText(path)
        self.engine = InferenceEngine(Path(path))
        self.status_label.setText("Weights updated")
        self.info_label.setText("Weights directory changed.")

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_edit.text() or str(Path.cwd()))
        if not path:
            return
        self.output_dir = Path(path)
        self.output_edit.setText(path)

    def run_current_mode(self) -> None:
        self._stop_video_playback()
        if self.worker_running:
            QMessageBox.information(self, "Busy", "A worker is already running.")
            return
        mode = self.source_combo.currentText()
        if mode == "camera":
            QMessageBox.information(self, "Camera mode", "Use Start Camera for live inference.")
            return
        if not self.selected_paths:
            QMessageBox.information(self, "No source", "Choose media or drop it onto the window first.")
            return

        self.stop_requested = False
        self.worker_running = True
        self._set_status("Running", "Preparing inference...")
        self._set_progress(0.0, "Starting", "FPS: -")
        self._sync_action_state()

        worker = {
            "image": self._run_image_worker,
            "video": self._run_video_worker,
        }[mode]
        self._start_thread(worker)

    def _start_thread(self, target) -> None:
        signals = WorkerSignals()
        signals.status_changed.connect(self._set_status)
        signals.progress_changed.connect(self._set_progress)
        signals.result_ready.connect(self._apply_result)
        signals.preview_ready.connect(self._update_previews)
        signals.live_meta.connect(self._update_live_meta)
        signals.camera_bootstrap.connect(self._show_camera_bootstrap)
        signals.completed.connect(self._finish_worker)
        signals.error.connect(self._show_error)
        thread = threading.Thread(target=target, args=(signals,), daemon=True)
        thread.start()

    def _run_image_worker(self, signals: WorkerSignals) -> None:
        try:
            task_key = self.task_combo.currentText()
            model_name = self.model_combo.currentText()
            self.engine.reset_tracker()
            signals.status_changed.emit("Running", f"Processing image with {model_name}...")
            result = self.engine.run(task_key, model_name, self.current_input_image.copy())
            signals.result_ready.emit(result)
            signals.progress_changed.emit(100.0, "1/1", "FPS: -")
            signals.completed.emit("")
        except Exception as exc:  # noqa: BLE001
            signals.error.emit(str(exc))

    def _run_video_worker(self, signals: WorkerSignals) -> None:
        input_path = self.selected_paths[0]
        task_key = self.task_combo.currentText()
        model_name = self.model_combo.currentText()
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            signals.error.emit(f"Could not open video: {input_path}")
            return

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        output_dir = self._resolve_output_dir()
        output_path = output_dir / f"{input_path.stem}_{task_key}.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        self.engine.reset_tracker()
        signals.status_changed.emit("Running", f"Processing video to {output_path.name}...")
        started = time.perf_counter()
        preview_stride = max(frame_count // 30, 1) if frame_count > 0 else 5

        try:
            index = 0
            while not self.stop_requested:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                result = self.engine.run(task_key, model_name, frame)
                writer.write(result.image if result.image is not None else frame)
                if index % preview_stride == 0:
                    preview = result.image.copy() if result.image is not None else frame.copy()
                    signals.preview_ready.emit(frame.copy(), preview)
                elapsed = max(time.perf_counter() - started, 1e-6)
                fps_text = f"FPS: {(index + 1) / elapsed:.2f}"
                percent = ((index + 1) / frame_count) * 100.0 if frame_count > 0 else 0.0
                signals.progress_changed.emit(percent, self._progress_text(index + 1, frame_count), fps_text)
                index += 1
        except Exception as exc:  # noqa: BLE001
            writer.release()
            capture.release()
            signals.error.emit(str(exc))
            return

        writer.release()
        capture.release()
        self.last_result_video_path = output_path
        summary = f"Saved video result to {output_path}"
        if self.stop_requested:
            summary = f"Stopped early. Partial result saved to {output_path}"
        signals.completed.emit(summary)

    def toggle_camera(self) -> None:
        self._stop_video_playback()
        if self.source_combo.currentText() != "camera":
            return
        if self.camera_running:
            self._stop_camera()
            return
        self.stop_requested = False
        camera_index = self.camera_spin.value()
        self.camera_capture = self._open_camera_capture(camera_index)
        if not self.camera_capture.isOpened():
            self.camera_capture = None
            self._show_error(f"Could not open camera device {camera_index}.")
            return
        ok, first_frame = self.camera_capture.read()
        if not ok or first_frame is None:
            self.camera_capture.release()
            self.camera_capture = None
            self._show_error(f"Camera device {camera_index} opened, but no frame was received.")
            return

        self.camera_running = True
        self.worker_running = True
        self.engine.reset_tracker()
        self._set_status("Camera", f"Live inference started on device {camera_index}.")
        self._set_progress(0.0, "Live", "FPS: -")
        self._show_camera_bootstrap(first_frame.copy())
        self._sync_action_state()
        self._start_thread(self._camera_loop)

    def _camera_loop(self, signals: WorkerSignals) -> None:
        task_key = self.task_combo.currentText()
        model_name = self.model_combo.currentText()
        started = time.perf_counter()
        frames = 0
        while self.camera_running and self.camera_capture is not None and not self.stop_requested:
            ok, frame = self.camera_capture.read()
            if not ok or frame is None:
                break
            try:
                result = self.engine.run(task_key, model_name, frame)
            except Exception as exc:  # noqa: BLE001
                signals.error.emit(str(exc))
                return
            frames += 1
            elapsed = max(time.perf_counter() - started, 1e-6)
            fps_text = f"FPS: {frames / elapsed:.2f}"
            result_frame = result.image.copy() if result.image is not None else frame.copy()
            self._overlay_frame_meta(result_frame, fps_text)
            signals.preview_ready.emit(frame.copy(), result_frame)
            signals.live_meta.emit(result.summary, fps_text)
            time.sleep(0.01)
        signals.completed.emit("")

    def _stop_camera(self) -> None:
        self.camera_running = False
        self.stop_requested = False
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None
        if self.source_combo.currentText() == "camera":
            self.status_label.setText("Ready")
            self.info_label.setText("Camera stopped.")
        self.progress_label.setText("Idle")
        self.fps_label.setText("FPS: -")
        self.worker_running = False
        self._sync_action_state()

    def request_stop(self) -> None:
        self.stop_requested = True
        self.info_label.setText("Stop requested. Finishing current frame...")

    def save_result(self) -> None:
        if self.source_combo.currentText() == "video" and self.last_result_video_path and self.last_result_video_path.exists():
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save processed video",
                str(self.last_result_video_path),
                "MP4 Video (*.mp4);;AVI Video (*.avi);;All files (*.*)",
            )
            if not path:
                return
            shutil.copyfile(self.last_result_video_path, path)
            self.status_label.setText("Saved")
            self.info_label.setText(f"Saved processed video to {path}")
            return

        if self.current_result_image is None:
            QMessageBox.information(self, "No result", "Run inference before saving.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save result image",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not path:
            return
        suffix = Path(path).suffix.lower() or ".png"
        ext = suffix if suffix in {".png", ".jpg", ".jpeg", ".bmp"} else ".png"
        ok, encoded = cv2.imencode(ext, self.current_result_image)
        if not ok:
            self._show_error("Could not encode the result image.")
            return
        encoded.tofile(path)
        self.status_label.setText("Saved")
        self.info_label.setText(f"Saved current preview to {path}")

    def _resolve_output_dir(self) -> Path:
        if self.output_edit.text():
            output_dir = Path(self.output_edit.text())
        else:
            base = self.selected_paths[0].parent if self.selected_paths else Path.cwd()
            output_dir = base / "python_onnx_outputs"
            self.output_edit.setText(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        return output_dir

    def _apply_result(self, result: InferenceResult) -> None:
        self._stop_video_playback()
        if result.image is None:
            self.status_label.setText("Failed")
            self.info_label.setText(result.summary or "Inference returned no image.")
            self.worker_running = False
            self._sync_action_state()
            return
        self.current_result_image = result.image
        self.result_card.set_image(result.image)
        if self.current_input_image is not None:
            self.original_card.set_image(self.current_input_image)
        self.status_label.setText("Completed")
        self.info_label.setText(result.summary)
        self.fps_label.setText("FPS: -")
        self._sync_action_state()

    def _update_previews(self, original: np.ndarray, result: np.ndarray) -> None:
        self.current_input_image = original
        self.current_result_image = result
        self.original_card.set_image(original)
        self.result_card.set_image(result)
        self._sync_action_state()

    def _update_live_meta(self, text: str, fps_text: str) -> None:
        self.info_label.setText(text)
        self.progress_label.setText("Live")
        self.fps_label.setText(fps_text)

    def _show_camera_bootstrap(self, frame: np.ndarray) -> None:
        self.current_input_image = frame
        self.original_card.set_image(frame)
        bootstrap = frame.copy()
        self._overlay_frame_meta(bootstrap, "Camera connected")
        self.current_result_image = bootstrap
        self.result_card.set_image(bootstrap)
        self.info_label.setText("Camera connected. Waiting for first inference result...")

    def _finish_worker(self, summary: str) -> None:
        if self.camera_running:
            self._stop_camera()
            return
        self.worker_running = False
        self.stop_requested = False
        if summary:
            self.status_label.setText("Completed")
            self.info_label.setText(summary)
            self.progress_bar.setValue(100)
        if self.source_combo.currentText() == "video" and self.last_result_video_path and self.last_result_video_path.exists():
            self._start_video_playback(self.selected_paths[0], self.last_result_video_path)
        self._sync_action_state()

    def _set_status(self, status: str, info: str) -> None:
        self.status_label.setText(status)
        self.info_label.setText(info)

    def _set_progress(self, value: float, text: str, fps_text: str) -> None:
        self.progress_bar.setValue(max(0, min(100, int(value))))
        self.progress_label.setText(text)
        self.fps_label.setText(fps_text)

    def _show_error(self, detail: str) -> None:
        self.worker_running = False
        self._stop_video_playback()
        if self.camera_running:
            self._stop_camera()
        self.status_label.setText("Error")
        self.info_label.setText(detail)
        self._sync_action_state()
        QMessageBox.critical(self, "Inference failed", detail)

    def _start_video_playback(self, original_path: Path, result_path: Path) -> None:
        self._stop_video_playback()
        self.original_playback_capture = cv2.VideoCapture(str(original_path))
        self.result_playback_capture = cv2.VideoCapture(str(result_path))
        if not self.original_playback_capture.isOpened() or not self.result_playback_capture.isOpened():
            self._stop_video_playback()
            return
        fps = self.result_playback_capture.get(cv2.CAP_PROP_FPS) or self.original_playback_capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.playback_timer.start(max(20, int(1000 / fps)))
        self._advance_video_playback()

    def _advance_video_playback(self) -> None:
        if self.source_combo.currentText() != "video" or self.camera_running:
            self._stop_video_playback()
            return
        if self.original_playback_capture is None or self.result_playback_capture is None:
            return
        ok_original, original = self.original_playback_capture.read()
        ok_result, result = self.result_playback_capture.read()
        if not ok_original or original is None:
            self.original_playback_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_original, original = self.original_playback_capture.read()
        if not ok_result or result is None:
            self.result_playback_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_result, result = self.result_playback_capture.read()
        if ok_original and original is not None:
            self.current_input_image = original
            self.original_card.set_image(original)
        if ok_result and result is not None:
            self.current_result_image = result
            self.result_card.set_image(result)

    def _stop_video_playback(self) -> None:
        self.playback_timer.stop()
        if self.original_playback_capture is not None:
            self.original_playback_capture.release()
            self.original_playback_capture = None
        if self.result_playback_capture is not None:
            self.result_playback_capture.release()
            self.result_playback_capture = None

    @staticmethod
    def _overlay_frame_meta(image: np.ndarray, text: str) -> None:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x0, y0 = 12, 12
        cv2.rectangle(image, (x0, y0), (x0 + tw + 16, y0 + th + baseline + 16), (16, 28, 46), -1)
        cv2.putText(image, text, (x0 + 8, y0 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 247, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _open_camera_capture(camera_index: int) -> cv2.VideoCapture:
        if sys.platform.startswith("win"):
            capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(camera_index)

    @staticmethod
    def _progress_text(current: int, total: int) -> str:
        if total <= 0:
            return f"Processed {current}"
        return f"Processed {current}/{total}"

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_requested = True
        self._stop_video_playback()
        self._stop_camera()
        super().closeEvent(event)


def run() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = VisionDesktopApp()
    window.show()
    app.exec_()
