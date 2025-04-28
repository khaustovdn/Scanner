import sys
import cv2
import pickle
import numpy as np
import face_recognition
from collections import namedtuple
import uuid
import queue
import threading
import logging
import signal
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    List,
    Dict,
    Tuple,
    Optional,
    Callable,
    Iterator
)
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import shlex

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QFormLayout,
    QGroupBox
)
from PySide6.QtCore import (
    Qt,
    QObject,
    Signal,
    SignalInstance,
    QTimer,
    QMetaObject
)
from PySide6.QtGui import (
    QPainter,
    QShowEvent
)
from PySide6.QtCharts import (
    QChart,
    QChartView,
    QBarSet,
    QBarSeries,
    QValueAxis,
    QBarCategoryAxis
)
from dependency_injector import containers, providers

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


class Config:
    ENCODINGS_DIR = Path("poll_face_encodings")
    IMAGE_DIR = Path("faces")

    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    CAMERA_FPS = 60
    SCALE = 0.5

    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720

    DET_MODEL = "hog"
    ENC_MODEL = "large"
    JITTERS = 10
    TOLERANCE = 0.5

    SHARP_TH = 80
    BRIGHTNESS_TH = 40
    CONTRAST_TH = 30

    CENTER_TH = 0.1
    SIZE_TH = 0.3
    ANGLE_TH = 15

    QUEUE_MAX = 5
    WAIT = 0.05

    REQ_LMS = {
        "left_eye", "right_eye", "nose_tip",
        "top_lip", "bottom_lip", "chin",
        "left_eyebrow", "right_eyebrow",
    }

    BOX_COLOR = (0, 255, 0)
    STATUS_COLOR = (0, 255, 255)
    BOX_THICKNESS_SCALE = 0.002
    FONT_SCALE_DIVISOR = 1000
    FONT_THICKNESS_FACTOR = 2

    @staticmethod
    def camera_settings(cap: cv2.VideoCapture) -> None:
        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            Config.CAMERA_WIDTH
        )
        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            Config.CAMERA_HEIGHT
        )
        cap.set(
            cv2.CAP_PROP_FPS,
            Config.CAMERA_FPS
        )
        cap.set(
            cv2.CAP_PROP_AUTOFOCUS,
            0
        )
        cap.set(
            cv2.CAP_PROP_FOCUS,
            50
        )


class FrameCapture(ABC):
    @abstractmethod
    def read(self) -> Optional[NDArray[np.uint8]]:
        pass

    @abstractmethod
    def release(self) -> None:
        pass


class FaceLocator(ABC):
    @abstractmethod
    def locate_faces(
        self,
        frame: NDArray[np.uint8]
    ) -> List[Tuple[int, int, int, int]]:
        pass

    @abstractmethod
    def find_landmarks(
        self,
        img: NDArray[np.uint8],
        locations: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[dict]:
        pass


class FeatureEncoder(ABC):
    @abstractmethod
    def generate_encodings(
        self,
        img: NDArray[np.uint8]
    ) -> List[NDArray[np.float64]]:
        pass

    @abstractmethod
    def match_features(
        self,
        known: List[NDArray[np.float64]],
        unknown: NDArray[np.float64],
        tolerance: float
    ) -> List[bool]:
        pass


class FaceStorage(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def add_entry(
        self,
        encoding: NDArray[np.float64],
    ) -> Optional[str]:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[dict]:
        pass


class DisplaySystem(ABC):
    @abstractmethod
    def update_display(
        self,
        frame: NDArray[np.uint8],
        faces: List[Tuple[int, int, int, int]],
        status: str
    ) -> None:
        pass


class FaceDatabase(FaceStorage):
    def __init__(
        self,
        encoder: FeatureEncoder,
        path: Path
    ) -> None:
        self._encoder = encoder
        self._storage_path = path
        self._lock = threading.Lock()
        self._entries: List[dict] = []
        self.load()

    def __iter__(self) -> Iterator[dict]:
        return iter(self._entries)

    def load(self) -> None:
        try:
            if not self._storage_path.exists():
                self._entries = []
                logging.info(
                    "База данных не найдена - создание новой"
                )
                return

            with self._storage_path.open("rb") as f:
                raw_data = pickle.load(f)

            if isinstance(
                raw_data,
                list
            ):
                self._handle_list_data(raw_data)

            elif isinstance(
                raw_data,
                dict
            ):
                self._handle_dict_data(raw_data)
            else:
                self._entries = []

            logging.info(
                f"Успешно загружено {len(self._entries)} "
                f"кодировок лиц"
            )

        except Exception as e:
            logging.error(f"Ошибка загрузки базы данных: {str(e)}")
            self._entries = []
            self._backup_corrupted_db()

    def _handle_list_data(
        self,
        data: list
    ) -> None:
        if data and isinstance(data[0], dict):
            self._entries = data
        else:
            self._entries = [
                {
                    "id": str(uuid.uuid4()),
                    "encoding": enc,
                    "path": "",
                    "timestamp": datetime.now().isoformat()
                }
                for enc in data
            ]

    def _handle_dict_data(
        self,
        data: dict
    ) -> None:
        self._entries = [
            {
                "id": str(uuid.uuid4()),
                "encoding": data.get('enc'),
                "path": data.get('path', ""),
                "timestamp": datetime.now().isoformat()
            }
        ]

    def _backup_corrupted_db(self) -> None:
        try:
            if self._storage_path.exists():
                backup_path = self._storage_path.with_suffix('.bak')
                self._storage_path.rename(backup_path)
                logging.warning(
                    f"Создана резервная копия поврежденной базы: {backup_path}"
                )
        except Exception as backup_error:
            logging.error(
                f"Ошибка создания резервной копии: {str(backup_error)}"
            )

    def save(self) -> None:
        with self._lock:
            try:
                temp_path = self._storage_path.with_suffix('.tmp')

                with temp_path.open("wb") as f:
                    pickle.dump(self._entries, f)

                temp_path.replace(self._storage_path)
                logging.info(
                    f"Сохранено {len(self._entries)} кодировок"
                )
            except Exception as e:
                logging.error(
                    f"Ошибка сохранения: {str(e)}"
                )

    def add_entry(
        self,
        encoding: NDArray[np.float64],
    ) -> Optional[str]:
        with self._lock:
            existing_id = self._find_existing_entry(encoding)
            if existing_id:
                return existing_id

            return self._create_new_entry(encoding)

    def _find_existing_entry(
        self,
        encoding: NDArray[np.float64]
    ) -> Optional[str]:
        for entry in self._entries:
            matches = self._encoder.match_features(
                known=[entry["encoding"]],
                unknown=encoding,
                tolerance=Config.TOLERANCE
            )
            if matches and matches[0]:
                logging.info(
                    f"Лицо уже существует в базе: {entry['id']}"
                )
                return entry["id"]
        return None

    def _create_new_entry(
        self,
        encoding: NDArray[np.float64],
    ) -> str:
        new_id = str(uuid.uuid4())
        new_entry = {
            "id": new_id,
            "encoding": encoding,
            "timestamp": datetime.now().isoformat()
        }
        self._entries.append(new_entry)
        logging.info(
            f"Добавлена новая запись: {new_id}"
        )
        return new_id


class FrameProcessor:
    def __init__(
        self,
        locator: FaceLocator
    ) -> None:
        self.locator = locator

    def is_centered(
        self,
        centroid: Tuple[int, int],
        frame_dims: Tuple[int, int]
    ) -> bool:
        frame_w, frame_h = frame_dims
        dx = abs(centroid[0] - frame_w / 2) / frame_w
        dy = abs(centroid[1] - frame_h / 2) / frame_h
        return (
            dx <= Config.CENTER_TH
            and dy <= Config.CENTER_TH
        )

    @staticmethod
    def is_sharp(image: NDArray[np.uint8]) -> bool:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
        laplacian_var = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
        return laplacian_var >= Config.SHARP_TH

    @staticmethod
    def is_bright(image: NDArray[np.uint8]) -> bool:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
        return gray.mean() >= Config.BRIGHTNESS_TH

    @staticmethod
    def has_contrast(image: NDArray[np.uint8]) -> bool:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
        return gray.std() >= Config.CONTRAST_TH

    @staticmethod
    def has_sufficient_size(
        location: Tuple[int, int, int, int],
        frame_dims: Tuple[int, int]
    ) -> bool:
        top, right, bottom, left = location
        width_ratio = (right - left) / frame_dims[0]
        height_ratio = (bottom - top) / frame_dims[1]
        return (
            width_ratio >= Config.SIZE_TH
            and height_ratio >= Config.SIZE_TH
        )

    def align_face(
        self,
        image: NDArray[np.uint8]
    ) -> Optional[NDArray[np.uint8]]:
        landmarks = self.locator.find_landmarks(image)

        if not landmarks or not Config.REQ_LMS.issubset(landmarks[0]):
            logging.warning(
                "Недостаточно ориентиров для выравнивания"
            )
            return None

        left_eye = np.mean(
            landmarks[0]["left_eye"],
            axis=0
        )
        right_eye = np.mean(
            landmarks[0]["right_eye"],
            axis=0
        )

        angle = np.degrees(
            np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            )
        )

        if abs(angle) > Config.ANGLE_TH:
            return None

        center = (
            int((left_eye[0] + right_eye[0]) / 2),
            int((left_eye[1] + right_eye[1]) / 2)
        )

        rot_mat = cv2.getRotationMatrix2D(
            center=center,
            angle=angle,
            scale=1.0
        )

        aligned = cv2.warpAffine(
            src=image,
            M=rot_mat,
            dsize=image.shape[1::-1],
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return aligned.astype(np.uint8)


class CameraCapture(FrameCapture):
    def __init__(
        self,
        scale: float = Config.SCALE
    ) -> None:
        self._device = cv2.VideoCapture(0)

        if not self._device.isOpened():
            raise RuntimeError(
                "Ошибка инициализации камеры"
            )

        Config.camera_settings(self._device)

        actual_width = int(
            self._device.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
        actual_height = int(
            self._device.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        logging.info(
            f"Камера настроена на разрешение: "
            f"{actual_width}x{actual_height}"
        )

        self._scaling = scale
        self._original_size = (
            actual_width,
            actual_height
        )
        self._target_size = (
            int(actual_width * scale),
            int(actual_height * scale)
        )

    def read(self) -> Optional[NDArray[np.uint8]]:
        success, frame = self._device.read()

        if not success or frame is None:
            logging.warning(
                "Не удалось получить кадр с камеры"
            )
            return None

        resized = cv2.resize(
            src=frame,
            dsize=None,
            fx=self._scaling,
            fy=self._scaling,
            interpolation=cv2.INTER_AREA
        )

        return resized.astype(np.uint8)

    def release(self) -> None:
        if self._device.isOpened():
            self._device.release()
            logging.info(
                "Ресурсы камеры освобождены"
            )

    def __enter__(self):
        return self

    def __exit__(
        self,
        *_
    ):
        self.release()


class RecognitionPipeline:
    def __init__(
        self,
        capture: FrameCapture,
        detector: FaceLocator,
        encoder: FeatureEncoder,
        database: FaceStorage,
        display: DisplaySystem,
    ) -> None:
        self.capture = capture
        self.detector = detector
        self.encoder = encoder
        self.database = database
        self.display = display
        self.frame_processor = FrameProcessor(detector)
        self._frame_queue = queue.Queue(Config.QUEUE_MAX)
        self._display_queue = queue.Queue(Config.QUEUE_MAX)
        self._running = True

        self._process_thread = threading.Thread(
            target=self._process_frames,
            daemon=True
        )
        self._process_thread.start()

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(
        self,
        *_: Any
    ) -> None:
        logging.info(
            "Инициируется завершение работы..."
        )
        self._running = False

    def _process_frames(self) -> None:
        while self._running or not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get(timeout=Config.WAIT)
                self._analyze_frame(frame)
                self._frame_queue.task_done()
            except queue.Empty:
                continue

    def _analyze_frame(
        self,
        frame: NDArray[np.uint8]
    ) -> None:
        face_locations = self.detector.locate_faces(frame)
        for loc in face_locations:
            self._process_face(
                frame,
                loc
            )
        self._queue_display(
            frame,
            face_locations
        )

    def _process_face(
        self,
        frame: NDArray[np.uint8],
        location: Tuple[int, int, int, int]
    ) -> None:
        if not self._valid_face(
            frame,
            location
        ):
            return

        aligned = self._align_face(
            frame,
            location
        )
        if aligned is None:
            logging.warning("Ошибка выравнивания лица")
            return

        encodings = self.encoder.generate_encodings(aligned)
        if not encodings:
            return

        self._handle_encoding(
            encodings[0],
            frame,
            location
        )

    def _valid_face(
        self,
        frame: NDArray[np.uint8],
        location: Tuple[int, int, int, int]
    ) -> bool:
        height, width = frame.shape[:2]
        return self.frame_processor.has_sufficient_size(
            location=location,
            frame_dims=(
                width,
                height
            )
        )

    def _align_face(
        self,
        frame: NDArray[np.uint8],
        location: Tuple[int, int, int, int]
    ) -> Optional[NDArray[np.uint8]]:
        top, right, bottom, left = location
        face_region = frame[top:bottom, left:right]
        return self.frame_processor.align_face(face_region)

    def _handle_encoding(
        self,
        encoding: NDArray[np.float64],
        frame: NDArray[np.uint8],
        location: Tuple[int, int, int, int]
    ) -> None:
        known = [entry["encoding"] for entry in self.database]
        matches = self.encoder.match_features(
            known=known,
            unknown=encoding,
            tolerance=Config.TOLERANCE
        )

        if any(matches):
            self._log_match(matches)
        elif self._quality_check(
            frame,
            location
        ):
            self._register_new_face(
                encoding
            )

    def _log_match(
        self,
        matches: List[bool]
    ) -> None:
        match_idx = matches.index(True)
        matched_id = list(self.database)[match_idx]["id"]
        logging.info(
            "Распознано лицо: %s",
            matched_id
        )

    def _quality_check(
        self,
        frame: NDArray[np.uint8],
        location: Tuple[int, int, int, int]
    ) -> bool:
        center_x = (location[3] + location[1]) // 2
        center_y = (location[0] + location[2]) // 2
        height, width = frame.shape[:2]

        return (
            self.frame_processor.is_centered(
                centroid=(
                    center_x,
                    center_y
                ),
                frame_dims=(
                    width,
                    height
                )
            )
            and self.frame_processor.is_sharp(
                frame[
                    location[0]:location[2],
                    location[3]:location[1]
                ]
            )
            and self.frame_processor.is_bright(
                frame[
                    location[0]:location[2],
                    location[3]:location[1]
                ]
            )
            and self.frame_processor.has_contrast(
                frame[
                    location[0]:location[2],
                    location[3]:location[1]
                ]
            )
        )

    def _register_new_face(
        self,
        encoding: NDArray[np.float64]
    ) -> None:

        if new_id := self.database.add_entry(
            encoding,
        ):
            self.database.save()
            logging.info(
                "Зарегистрировано новое лицо: %s",
                new_id
            )

    def _queue_display(
        self,
        frame: NDArray[np.uint8],
        faces: List[Tuple[int, int, int, int]]
    ) -> None:
        try:
            self._display_queue.put(
                item=(
                    frame,
                    faces
                ),
                block=False
            )
        except queue.Full:
            logging.debug(
                "Очередь отображения переполнена"
            )

    def _cleanup(self) -> None:
        try:
            self.capture.release()
            if isinstance(
                self.display,
                DisplayAdapter
            ):
                cv2.destroyAllWindows()
            logging.info(
                "Ресурсы освобождены"
            )
        except Exception as e:
            logging.error(
                "Ошибка очистки: %s",
                e
            )

    def execute(self) -> None:
        try:
            while self._running:
                if (frame := self.capture.read()) is None:
                    break

                try:
                    self._frame_queue.put(
                        frame,
                        block=False
                    )
                except queue.Full:
                    logging.debug(
                        "Очередь кадров переполнена"
                    )

                self._update_interface()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running = False
                    break
        finally:
            self._shutdown()

    def _update_interface(self) -> None:
        try:
            frame, faces = self._display_queue.get_nowait()
            status = (
                f"Обнаружено: {len(faces)} лиц"
                if faces else "Лица не обнаружены"
            )
            self.display.update_display(
                frame=frame,
                faces=faces,
                status=status
            )
        except queue.Empty:
            pass

    def _shutdown(self) -> None:
        self._running = False
        self._process_thread.join(timeout=2.0)
        self._cleanup()
        logging.info(
            "Система завершила работу"
        )


class DetectionEngine(FaceLocator):
    def __init__(
        self,
        model: str = Config.DET_MODEL
    ) -> None:
        self._model = model

    def locate_faces(
        self,
        frame: NDArray[np.uint8]
    ) -> List[Tuple[int, int, int, int]]:
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        return face_recognition.face_locations(
            rgb_frame,
            model=self._model
        )

    def find_landmarks(
        self,
        img: NDArray[np.uint8],
        locations: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[dict]:
        return face_recognition.face_landmarks(
            img,
            face_locations=locations
        )


class EncodingEngine(FeatureEncoder):
    def __init__(
        self,
        model: str = Config.ENC_MODEL,
        jitters: int = Config.JITTERS
    ) -> None:
        self._model = model
        self._jitter = jitters

    def generate_encodings(
        self,
        img: NDArray[np.uint8]
    ) -> List[NDArray[np.float64]]:
        return face_recognition.face_encodings(
            img,
            model=self._model,
            num_jitters=self._jitter
        )

    def match_features(
        self,
        known: List[NDArray[np.float64]],
        unknown: NDArray[np.float64],
        tolerance: float
    ) -> List[bool]:
        return face_recognition.compare_faces(
            known_face_encodings=known,
            face_encoding_to_check=unknown,
            tolerance=tolerance
        )


class DisplayAdapter(DisplaySystem):
    def __init__(self) -> None:
        self._window_name = "Система распознавания лиц"
        cv2.namedWindow(
            winname=self._window_name,
            flags=cv2.WINDOW_NORMAL
        )
        cv2.resizeWindow(
            winname=self._window_name,
            width=Config.DISPLAY_WIDTH,
            height=Config.DISPLAY_HEIGHT
        )

    def update_display(
        self,
        frame: NDArray[np.uint8],
        faces: List[Tuple[int, int, int, int]],
        status: str
    ) -> None:
        try:
            display_frame = cv2.resize(
                src=frame,
                dsize=(
                    Config.DISPLAY_WIDTH,
                    Config.DISPLAY_HEIGHT
                ),
                interpolation=cv2.INTER_CUBIC
            ).astype(np.uint8)

            font_params = self._calculate_font_params()
            self._draw_face_boxes(
                display_frame,
                faces
            )
            self._add_status_text(
                frame=display_frame,
                status=status,
                font_scale=font_params['scale'],
                thickness=font_params['thickness']
            )

            cv2.imshow(
                self._window_name,
                display_frame
            )

        except Exception as e:
            logging.error(
                "Ошибка отображения: %s",
                e
            )

    def _calculate_font_params(self) -> dict:
        font_scale = (
            Config.DISPLAY_WIDTH / Config.FONT_SCALE_DIVISOR
        )
        return {
            'scale': font_scale,
            'thickness': max(
                1,
                int(font_scale * Config.FONT_THICKNESS_FACTOR)
            )
        }

    def _draw_face_boxes(
        self,
        frame: NDArray[np.uint8],
        faces: List[Tuple[int, int, int, int]]
    ) -> None:
        for face in faces:
            scaled = self._scale_face_coordinates(
                face_coords=face,
                frame_height=frame.shape[0],
                frame_width=frame.shape[1]
            )
            cv2.rectangle(
                img=frame,
                pt1=(scaled.left, scaled.top),
                pt2=(scaled.right, scaled.bottom),
                color=Config.BOX_COLOR,
                thickness=scaled.thickness
            )

    def _scale_face_coordinates(
        self,
        face_coords: Tuple[int, int, int, int],
        frame_height: int,
        frame_width: int
    ) -> Any:
        ScaleResult = namedtuple(
            'ScaleResult',
            [
                'top',
                'right',
                'bottom',
                'left',
                'thickness'
            ]
        )
        top, right, bottom, left = face_coords

        scale_x = Config.DISPLAY_WIDTH / frame_width
        scale_y = Config.DISPLAY_HEIGHT / frame_height

        return ScaleResult(
            top=int(top * scale_y),
            right=int(right * scale_x),
            bottom=int(bottom * scale_y),
            left=int(left * scale_x),
            thickness=max(
                1,
                int(
                    Config.BOX_THICKNESS_SCALE * Config.DISPLAY_WIDTH
                )
            )
        )

    def _add_status_text(
        self,
        frame: NDArray[np.uint8],
        status: str,
        font_scale: float,
        thickness: int
    ) -> None:
        cv2.putText(
            img=frame,
            text=status,
            org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=Config.STATUS_COLOR,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

    def close(self) -> None:
        try:
            if cv2.getWindowProperty(
                self._window_name,
                cv2.WND_PROP_VISIBLE
            ) >= 0:
                cv2.destroyWindow(self._window_name)
                logging.debug(
                    "Окно отображения закрыто"
                )
        except Exception as e:
            logging.debug(
                "Ошибка закрытия окна: %s",
                e
            )


class ValidationError(Exception):
    def __init__(
        self,
        message: str,
        field: Optional[str] = None
    ) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


class Logger:
    LEVELS = [
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR"
    ]

    def __init__(self) -> None:
        self._level = "INFO"

    def configure(
        self,
        level: str
    ) -> None:
        if level in self.LEVELS:
            self._level = level

    def log(
        self,
        level: str,
        module: str,
        message: str,
        *args: Any
    ) -> None:
        if self.LEVELS.index(level) < self.LEVELS.index(self._level):
            return

        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        formatted_msg = message % args
        print(
            f"{timestamp} [{level}] [{module}] {formatted_msg}"
        )

    def debug(
        self,
        module: str,
        message: str,
        *args: Any
    ) -> None:
        self.log(
            "DEBUG",
            module,
            message,
            *args
        )

    def info(
        self,
        module: str,
        message: str,
        *args: Any
    ) -> None:
        self.log(
            "INFO",
            module,
            message,
            *args
        )

    def warning(
        self,
        module: str,
        message: str,
        *args: Any
    ) -> None:
        self.log(
            "WARNING",
            module,
            message,
            *args
        )

    def error(
        self,
        module: str,
        message: str,
        *args: Any
    ) -> None:
        self.log(
            "ERROR",
            module,
            message,
            *args
        )


class PollModel(QObject):
    data_changed = Signal()

    def __init__(
        self,
        title: str,
        options: List[str],
        logger: Logger,
        encoder: FeatureEncoder,
        poll_id: str
    ) -> None:
        super().__init__()
        self.logger = logger
        self.title = title
        self.poll_id = poll_id
        self._options: Dict[str, int] = {opt: 0 for opt in options}
        self.database = FaceDatabase(
            encoder=encoder,
            path=Config.ENCODINGS_DIR / f"{self.poll_id}.pkl"
        )
        self._cached_total = 0

    def add_vote(self, option: str) -> bool:
        if option not in self._options:
            self.logger.error(
                "PollModel",
                "Неверный вариант '%s' в опросе '%s'",
                option,
                self.title
            )
            return False

        self._options[option] += 1
        self._cached_total += 1
        self.data_changed.emit()
        return True

    def reset_votes(self) -> None:
        self._options = {k: 0 for k in self._options}
        self._reinit_database()
        self._cached_total = 0
        self.data_changed.emit()

    def update_options(
        self,
        new_options: List[str]
    ) -> None:
        self._options = {opt: 0 for opt in new_options}
        self._reinit_database()
        self._cached_total = 0
        self.data_changed.emit()

    def _reinit_database(self) -> None:
        self.database = FaceDatabase(
            encoder=EncodingEngine(),
            path=Config.ENCODINGS_DIR / f"{self.poll_id}.pkl"
        )

    @property
    def total_votes(self) -> int:
        return self._cached_total

    @property
    def options(self) -> Dict[str, int]:
        return self._options.copy()


class AppModel(QObject):
    polls_changed = Signal()
    admin_status_changed = Signal(bool)

    def __init__(
        self,
        logger: Logger
    ) -> None:
        super().__init__()
        self.logger = logger
        self._polls: List[PollModel] = []
        self._admin_logged_in: bool = False
        self._admin_credentials: Tuple[str, str] = ("admin", "admin")
        self.encoder = EncodingEngine()

    def create_poll(
        self,
        title: str,
        options: List[str]
    ) -> PollModel:
        existing_titles = [p.title for p in self._polls]

        self._validate_poll_title(
            title,
            existing_titles
        )
        self._validate_options(options)

        poll_id = str(uuid.uuid4())
        poll = PollModel(
            title=title,
            options=options,
            logger=self.logger,
            encoder=self.encoder,
            poll_id=poll_id
        )
        self._polls.append(poll)
        self.logger.info(
            "AppModel",
            "Создан опрос '%s' с вариантами: %s",
            title,
            options
        )
        self.polls_changed.emit()
        return poll

    def edit_poll(
        self,
        index: int,
        new_title: str,
        new_options: List[str]
    ) -> None:
        existing_titles = [
            p.title
            for p in self._polls
            if p != self._polls[index]
        ]

        self._validate_poll_title(
            new_title,
            existing_titles
        )
        self._validate_options(new_options)

        poll = self._polls[index]
        old_title = poll.title
        poll.title = new_title
        poll.update_options(new_options)
        self.logger.info(
            "AppModel",
            "Обновлен опрос '%s' -> '%s' с вариантами: %s",
            old_title,
            new_title,
            new_options
        )
        self.polls_changed.emit()

    def reset_poll(
        self,
        index: int
    ) -> None:
        self._polls[index].reset_votes()
        self.polls_changed.emit()

    def authenticate_admin(
        self,
        username: str,
        password: str
    ) -> bool:
        success = (
            username == self._admin_credentials[0]
            and password == self._admin_credentials[1]
        )
        self._admin_logged_in = success
        self.admin_status_changed.emit(success)
        return success

    @property
    def polls(self) -> List[PollModel]:
        return self._polls.copy()

    @property
    def is_admin(self) -> bool:
        return self._admin_logged_in

    @staticmethod
    def _validate_poll_title(
        title: str,
        existing_titles: List[str]
    ) -> None:
        if not title.strip():
            raise ValidationError(
                "Название опроса не может быть пустым",
                "title"
            )
        if title in existing_titles:
            raise ValidationError(
                "Название опроса должно быть уникальным",
                "title"
            )

    @staticmethod
    def _validate_options(options: List[str]) -> None:
        if len(options) < 2:
            raise ValidationError(
                "Требуется минимум 2 варианта",
                "options"
            )
        if any(not opt.strip() for opt in options):
            raise ValidationError(
                "Варианты не могут быть пустыми",
                "options"
            )
        if len(options) != len(set(options)):
            raise ValidationError(
                "Обнаружены дублирующиеся варианты",
                "options"
            )


class BaseView(QWidget):
    def __init__(
        self,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Dialog)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)


class MainView(QWidget):
    poll_selected = Signal(int)
    admin_login_requested = Signal()
    admin_logout_requested = Signal()
    add_poll_requested = Signal()
    edit_poll_requested = Signal(int)

    def __init__(
        self,
        logger: Logger
    ) -> None:
        super().__init__()
        self.logger = logger
        self._init_ui()
        self.setMinimumSize(400, 500)

    def _init_ui(self) -> None:
        self.setWindowTitle("Система голосования")
        main_layout = QVBoxLayout()

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._create_poll_list()
        self._create_user_group()
        self._create_admin_group()

        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.poll_list_group)
        main_layout.addWidget(self.user_group)
        main_layout.addWidget(self.admin_group)

        self.setLayout(main_layout)
        self.update_admin_status(False)

    def _create_poll_list(self) -> None:
        self.poll_list = QListWidget()
        self.poll_list.itemDoubleClicked.connect(
            lambda: self.poll_selected.emit(
                self.poll_list.currentRow()
            )
        )
        self.poll_list_group = QGroupBox("Доступные опросы")
        layout = QVBoxLayout()
        layout.addWidget(self.poll_list)
        self.poll_list_group.setLayout(layout)

    def _create_user_group(self) -> None:
        self.user_group = QGroupBox("Голосование")
        layout = QVBoxLayout()
        self.vote_btn = QPushButton("Голосовать")
        self.vote_btn.clicked.connect(
            lambda: self.poll_selected.emit(
                self.poll_list.currentRow()
            )
        )
        layout.addWidget(self.vote_btn)
        self.user_group.setLayout(layout)

    def _create_admin_group(self) -> None:
        self.admin_group = QGroupBox("Панель администратора")
        layout = QVBoxLayout()

        self.admin_login_btn = QPushButton("Вход")
        self.admin_login_btn.clicked.connect(
            self.admin_login_requested.emit
        )

        self.add_poll_btn = QPushButton("Создать опрос")
        self.add_poll_btn.clicked.connect(
            self.add_poll_requested.emit
        )
        self.add_poll_btn.setVisible(False)

        self.edit_poll_btn = QPushButton("Изменить опрос")
        self.edit_poll_btn.clicked.connect(self._on_edit_poll)
        self.edit_poll_btn.setVisible(False)

        self.results_btn = QPushButton("Результаты")
        self.results_btn.clicked.connect(self._show_results)
        self.results_btn.setVisible(False)

        self.logout_btn = QPushButton("Выход")
        self.logout_btn.clicked.connect(self._confirm_logout)
        self.logout_btn.setVisible(False)

        layout.addWidget(self.admin_login_btn)
        layout.addWidget(self.add_poll_btn)
        layout.addWidget(self.edit_poll_btn)
        layout.addWidget(self.results_btn)
        layout.addWidget(self.logout_btn)
        self.admin_group.setLayout(layout)

    def _on_edit_poll(self) -> None:
        index = self.poll_list.currentRow()
        if index >= 0:
            self.edit_poll_requested.emit(index)

    def _show_results(self) -> None:
        index = self.poll_list.currentRow()
        if index >= 0:
            self.poll_selected.emit(index)

    def _confirm_logout(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Подтверждение",
            "Вы уверены, что хотите выйти из режима администратора?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.admin_logout_requested.emit()

    def update_polls(
        self,
        polls: List[PollModel]
    ) -> None:
        current_titles = {
            self.poll_list.item(i).text()
            for i in range(self.poll_list.count())
        }
        new_titles = {poll.title for poll in polls}

        for title in current_titles - new_titles:
            items = self.poll_list.findItems(
                title,
                Qt.MatchFlag.MatchExactly
            )
            for item in items:
                self.poll_list.takeItem(self.poll_list.row(item))

        for poll in polls:
            if poll.title not in current_titles:
                item = QListWidgetItem(poll.title)
                self.poll_list.addItem(item)

    def update_admin_status(
        self,
        is_admin: bool
    ) -> None:
        status = "Режим администратора" if is_admin else "Режим пользователя"
        self.status_label.setText(status)
        self.user_group.setVisible(not is_admin)
        self.vote_btn.setVisible(not is_admin)
        self.add_poll_btn.setVisible(is_admin)
        self.edit_poll_btn.setVisible(is_admin)
        self.admin_login_btn.setVisible(not is_admin)
        self.logout_btn.setVisible(is_admin)


class AdminAuthView(BaseView):
    login_attempt = Signal(str, str)

    def __init__(
        self,
        logger: Logger
    ) -> None:
        super().__init__()
        self.logger = logger
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QFormLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Логин")

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("Пароль")

        self.login_btn = QPushButton("Войти")
        self.login_btn.clicked.connect(
            lambda: self.login_attempt.emit(
                self.username_input.text(),
                self.password_input.text()
            )
        )

        layout.addRow(
            "Логин:",
            self.username_input
        )
        layout.addRow(
            "Пароль:",
            self.password_input
        )
        layout.addWidget(self.login_btn)

        self.setLayout(layout)


class PollCreatorView(BaseView):
    create_poll = Signal(str, list)

    def __init__(
        self,
        logger: Logger
    ) -> None:
        super().__init__()
        self.logger = logger
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QFormLayout()

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Название опроса")

        self.options_input = QLineEdit()
        self.options_input.setPlaceholderText("Вариант1, Вариант2, ...")

        self.create_btn = QPushButton("Создать")
        self.create_btn.clicked.connect(self._on_create)

        layout.addRow(
            "Название:",
            self.title_input
        )
        layout.addRow(
            "Варианты (через запятую):",
            self.options_input
        )
        layout.addWidget(self.create_btn)

        self.setLayout(layout)

    def _on_create(self) -> None:
        title = self.title_input.text().strip()
        raw_options = self.options_input.text()

        try:
            options = [
                opt.strip()
                for opt in shlex.split(
                    raw_options,
                    posix=False
                )
                if opt.strip()
            ]
        except ValueError as e:
            QMessageBox.warning(
                self,
                "Ошибка",
                f"Некорректный формат вариантов: {str(e)}"
            )
            return

        self.create_poll.emit(
            title,
            options
        )


class PollEditorView(BaseView):
    update_poll = Signal(str, list)

    def __init__(
        self,
        poll: PollModel,
        logger: Logger
    ) -> None:
        super().__init__()
        self.logger = logger
        self.poll = poll
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QFormLayout()

        self.title_input = QLineEdit(self.poll.title)
        self.title_input.setPlaceholderText("Новое название опроса")

        self.options_input = QLineEdit(
            ", ".join(self.poll.options.keys())
        )
        self.options_input.setPlaceholderText(
            "Обновленные варианты (через запятую)"
        )

        self.update_btn = QPushButton("Обновить")
        self.update_btn.clicked.connect(self._on_update)

        layout.addRow(
            "Название:",
            self.title_input
        )
        layout.addRow(
            "Варианты:",
            self.options_input
        )
        layout.addWidget(self.update_btn)

        self.setLayout(layout)

    def _on_update(self) -> None:
        title = self.title_input.text().strip()
        raw_options = self.options_input.text()

        try:
            options = [
                opt.strip()
                for opt in shlex.split(
                    raw_options,
                    posix=False
                )
                if opt.strip()
            ]
        except ValueError as e:
            QMessageBox.warning(
                self,
                "Ошибка",
                f"Некорректный формат вариантов: {str(e)}"
            )
            return

        self.update_poll.emit(
            title,
            options
        )


class VotingView(BaseView):
    vote_submitted = Signal(str)

    def __init__(
        self,
        poll: PollModel,
        readonly: bool,
        logger: Logger
    ) -> None:
        super().__init__()
        self.poll = poll
        self.logger = logger
        self.readonly = readonly
        self._chart_initialized = False
        self._init_ui()
        self.setMinimumSize(800, 600)
        self.setWindowTitle(f"Результаты голосования - {self.poll.title}")

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        if self.readonly:
            self._create_chart(main_layout)

        self._create_options_list(main_layout)

        if not self.readonly:
            self._create_buttons(main_layout)

    def _create_chart(
        self,
        layout: QVBoxLayout
    ) -> None:
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.chart_view, 3)

    def _create_options_list(
        self,
        layout: QVBoxLayout
    ) -> None:
        self.options_list = QListWidget()
        mode = (
            QListWidget.SelectionMode.NoSelection
            if self.readonly
            else QListWidget.SelectionMode.SingleSelection
        )
        self.options_list.setSelectionMode(mode)
        layout.addWidget(self.options_list)

    def _create_buttons(
        self,
        layout: QVBoxLayout
    ) -> None:
        btn_layout = QHBoxLayout()
        self.vote_btn = QPushButton("Голосовать")
        self.vote_btn.clicked.connect(self._on_vote)
        btn_layout.addWidget(self.vote_btn)
        layout.addLayout(btn_layout)

    def showEvent(
        self,
        event: QShowEvent
    ) -> None:
        if self.readonly and not self._chart_initialized:
            self._init_chart()
            self._chart_initialized = True
        self._update_view()
        super().showEvent(event)

    def _init_chart(self) -> None:
        self.chart = QChart()
        self.chart.setBackgroundRoundness(10)
        self.chart_view.setChart(self.chart)

    def _update_chart(self) -> None:
        self.chart.removeAllSeries()
        for axis in self.chart.axes():
            self.chart.removeAxis(axis)

        bar_set = QBarSet("Голоса")
        values = list(self.poll.options.values())
        bar_set.append(values)

        series = QBarSeries()
        series.setBarWidth(0.6)
        series.append(bar_set)

        max_value = max(values) if values else 0
        upper = max(max_value, 1)

        categories = [
            f"{opt}\n({votes})"
            for opt, votes in self.poll.options.items()
        ]

        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        axis_x.setTitleText("Варианты голосования")

        axis_y = QValueAxis()
        axis_y.setTitleText("Количество голосов")
        axis_y.setRange(0, upper)
        axis_y.setTickInterval(1)

        self.chart.addSeries(series)
        self.chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)

    def _update_options(self) -> None:
        self.options_list.clear()
        max_votes = max(self.poll.options.values(), default=0)

        for option, votes in self.poll.options.items():
            text = (f"{option}: {votes} голосов" if self.readonly
                    else option)
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, option)

            if self.readonly and votes == max_votes:
                item.setBackground(Qt.GlobalColor.green)
                item.setForeground(Qt.GlobalColor.white)

            self.options_list.addItem(item)

    def _update_view(self) -> None:
        try:
            self.setUpdatesEnabled(False)
            if self.readonly:
                self._update_chart()
                self.chart.setTitle(
                    f"Результаты голосования - {self.poll.title}"
                )
            self._update_options()
        finally:
            self.setUpdatesEnabled(True)

    def _on_vote(self) -> None:
        self.vote_btn.setEnabled(False)
        try:
            self._process_vote()
        except ValidationError as e:
            QMessageBox.warning(self, "Ошибка", e.message)
        except Exception as e:
            self.logger.error(
                "VotingView",
                "Ошибка обработки голоса: %s",
                str(e)
            )
            QMessageBox.critical(
                self,
                "Ошибка",
                "Произошла непредвиденная ошибка"
            )
        finally:
            self.vote_btn.setEnabled(True)

    def _process_vote(self) -> None:
        option = self._get_selected_option()
        if not option:
            raise ValidationError(
                "Выберите вариант",
                "option"
            )

        with CameraCapture() as camera:
            frame = camera.read()
            if frame is None:
                raise ValidationError(
                    "Не удалось получить изображение"
                )

            detector = DetectionEngine()
            faces = detector.locate_faces(frame)
            if len(faces) != 1:
                raise ValidationError(
                    "Должно быть видно одно лицо"
                )

            top, right, bottom, left = faces[0]
            face_img = frame[top:bottom, left:right]

            processor = FrameProcessor(detector)
            aligned = processor.align_face(face_img)
            if aligned is None:
                raise ValidationError(
                    "Ошибка выравнивания лица"
                )

            encoder = EncodingEngine()
            encodings = encoder.generate_encodings(aligned)
            if not encodings:
                raise ValidationError(
                    "Ошибка генерации данных лица"
                )

            encoding = encodings[0]
            known = [e["encoding"] for e in self.poll.database]

            if encoder.match_features(known, encoding, Config.TOLERANCE):
                raise ValidationError(
                    "Вы уже голосовали в этом опросе"
                )

            self.poll.database.add_entry(encoding)
            self.poll.database.save()

            if not self.poll.add_vote(option):
                raise ValidationError(
                    "Ошибка сохранения голоса"
                )

            QMessageBox.information(
                self,
                "Успех",
                "Голос успешно зарегистрирован"
            )
            self.options_list.clearSelection()

    def _get_selected_option(self) -> Optional[str]:
        if item := self.options_list.currentItem():
            return item.text().split(":")[0].strip()
        return None


class BaseController(QObject):
    def __init__(
        self,
        parent: Optional[QObject] = None
    ) -> None:
        super().__init__(parent)
        self._connections: List[QMetaObject.Connection] = []

    def manage_connection(
        self,
        signal: SignalInstance,
        slot: Callable
    ) -> None:
        connection = signal.connect(slot)
        self._connections.append(connection)

    def cleanup(self) -> None:
        for conn in self._connections:
            QObject.disconnect(conn)
        self._connections.clear()


class Container(containers.DeclarativeContainer):
    logger = providers.Singleton(Logger)
    model = providers.Factory(AppModel, logger=logger)
    main_view = providers.Factory(MainView, logger=logger)
    auth_view = providers.Factory(AdminAuthView, logger=logger)
    poll_creator_view = providers.Factory(PollCreatorView, logger=logger)
    voting_view = providers.Factory(VotingView, logger=logger)
    poll_editor_view = providers.Factory(PollEditorView, logger=logger)


class AdminAuthController(BaseController):
    def __init__(
        self,
        model: AppModel,
        view: AdminAuthView
    ) -> None:
        super().__init__()
        self.model = model
        self.view = view
        self.manage_connection(
            self.view.login_attempt,
            self._authenticate
        )

    def _authenticate(
        self,
        username: str,
        password: str
    ) -> None:
        if self.model.authenticate_admin(
            username,
            password
        ):
            self.view.close()
            QMessageBox.information(
                self.view,
                "Success",
                "Logged in as administrator"
            )
        else:
            QMessageBox.warning(
                self.view,
                "Error",
                "Invalid administrator credentials"
            )


class PollCreatorController(BaseController):
    def __init__(
        self,
        model: AppModel,
        view: PollCreatorView
    ) -> None:
        super().__init__()
        self.model = model
        self.view = view
        self.manage_connection(
            self.view.create_poll,
            self._create_poll
        )

    def _create_poll(
        self,
        title: str,
        options: List[str]
    ) -> None:
        try:
            self.model.create_poll(
                title,
                options
            )
            self.view.close()
            QMessageBox.information(
                self.view,
                "Success",
                "Poll created successfully"
            )
        except ValidationError as e:
            QMessageBox.warning(
                self.view,
                "Error",
                e.message
            )


class PollEditorController(BaseController):
    def __init__(
        self,
        model: AppModel,
        view: PollEditorView,
        poll_index: int
    ) -> None:
        super().__init__()
        self.model = model
        self.view = view
        self.poll_index = poll_index
        self.manage_connection(
            self.view.update_poll,
            self._update_poll
        )

    def _update_poll(
        self,
        title: str,
        options: List[str]
    ) -> None:
        try:
            self.model.edit_poll(
                self.poll_index,
                title, options
            )
            self.view.close()
            QMessageBox.information(
                self.view,
                "Success",
                "Poll updated successfully"
            )
        except ValidationError as e:
            QMessageBox.warning(
                self.view,
                "Error",
                e.message
            )


class VotingController(BaseController):
    def __init__(
        self,
        poll: PollModel,
        view: VotingView,
        logger: Logger
    ) -> None:
        super().__init__(view)
        self.poll = poll
        self.view = view
        self.logger = logger
        self.manage_connection(
            self.view.vote_submitted,
            self._process_vote
        )
        self.manage_connection(
            self.poll.data_changed,
            self._update_view
        )
        QTimer.singleShot(
            0,
            self._update_view
        )

    def _update_view(self) -> None:
        self.view._update_view()

    def _process_vote(self, _: str) -> None:
        pass


class MainController(BaseController):
    def __init__(
        self,
        model: AppModel,
        view: MainView,
        auth_view_factory: Callable[[], AdminAuthView],
        poll_creator_factory: Callable[[], PollCreatorView],
        poll_editor_factory: Callable[[PollModel], PollEditorView],
        voting_view_factory: Callable[[PollModel, bool], VotingView]
    ) -> None:
        super().__init__()
        self.model = model
        self.view = view
        self._auth_view_factory = auth_view_factory
        self._poll_creator_factory = poll_creator_factory
        self._poll_editor_factory = poll_editor_factory
        self._voting_view_factory = voting_view_factory
        self._connect_signals()
        self.view.show()

    def _connect_signals(self) -> None:
        connections = [
            (
                self.view.poll_selected,
                self._show_voting
            ),
            (
                self.view.add_poll_requested,
                self._show_poll_creator
            ),
            (
                self.view.admin_login_requested,
                self._show_admin_login
            ),
            (
                self.view.admin_logout_requested,
                self._logout_admin
            ),
            (
                self.view.edit_poll_requested,
                self._show_poll_editor
            ),
            (
                self.model.polls_changed,
                self._update_polls_display
            ),
            (
                self.model.admin_status_changed,
                self._update_admin_ui
            ),
        ]
        for sig, handler in connections:
            self.manage_connection(
                sig,
                handler
            )

    def _show_voting(
        self,
        index: int
    ) -> None:
        if not 0 <= index < len(self.model.polls):
            QMessageBox.warning(
                self.view,
                "Ошибка",
                "Неверный опрос"
            )
            return

        poll = self.model.polls[index]
        view = self._voting_view_factory(
            poll,
            self.model.is_admin,
        )
        controller = VotingController(
            poll,
            view,
            self.model.logger
        )
        controller.setParent(view)
        view.show()

    def _show_admin_login(self) -> None:
        dialog = self._auth_view_factory()
        controller = AdminAuthController(self.model, dialog)
        controller.setParent(dialog)
        dialog.show()

    def _show_poll_creator(self) -> None:
        dialog = self._poll_creator_factory()
        controller = PollCreatorController(self.model, dialog)
        controller.setParent(dialog)
        dialog.show()

    def _show_poll_editor(self, index: int) -> None:
        if not 0 <= index < len(self.model.polls):
            QMessageBox.warning(self.view, "Error", "Invalid poll")
            return
        poll = self.model.polls[index]
        dialog = self._poll_editor_factory(poll)
        controller = PollEditorController(self.model, dialog, index)
        controller.setParent(dialog)
        dialog.show()

    def _logout_admin(self) -> None:
        self.model.authenticate_admin("", "")

    def _update_polls_display(self) -> None:
        self.view.update_polls(self.model.polls)

    def _update_admin_ui(self, is_admin: bool) -> None:
        self.view.update_admin_status(is_admin)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Config.ENCODINGS_DIR.mkdir(exist_ok=True, parents=True)
    container = Container()
    controller = MainController(
        model=container.model(),
        view=container.main_view(),
        auth_view_factory=container.auth_view,
        poll_creator_factory=container.poll_creator_view,
        poll_editor_factory=container.poll_editor_view,
        voting_view_factory=container.voting_view
    )
    sys.exit(app.exec())
