# -*- coding: utf-8 -*-

# region IMPORTS


import sys
import re
import shlex
from datetime import datetime
from typing import Any, List, Dict, Set, Tuple, Optional, Callable
from gettext import gettext as _

from PySide6.QtCore import (
    Signal,
    SignalInstance,
    QObject,
    Qt,
    QTimer,
    QMetaObject,
)
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QLineEdit,
    QMessageBox,
    QGroupBox,
    QFormLayout,
    QGraphicsDropShadowEffect,
)
from PySide6.QtGui import (
    QPainter,
    QPalette,
    QPen,
    QColor,
    QFont,
    QShowEvent,
)
from PySide6.QtCharts import (
    QChart,
    QChartView,
    QBarSet,
    QBarSeries,
    QBarCategoryAxis,
    QLegend,
    QValueAxis,
)
from dependency_injector import containers, providers
# endregion


# region UTILITIES


class DarkTheme:
    """Catppuccin Mocha dark theme with enhanced visual design system."""

    _CATPPUCCIN_PALETTE = {
        "base": "#1e1e2e",            # Base
        "text": "#cdd6f4",            # Text
        "disabled_text": "#6c7086",   # Overlay0
        "highlight": "#89b4fa",       # Blue
        "highlight_text": "#1e1e2e",  # Base
        "alternate_base": "#313244",  # Surface0
        "tooltip_base": "#585b70",    # Surface2
        "button": "#313244",          # Surface0
        "button_text": "#cdd6f4",     # Text
        "bright_text": "#f38ba8",     # Red
        "surface1": "#45475a",        # Surface1
        "overlay0": "#6c7086",        # Overlay0
        "graph_colors": [             # Catppuccin accent colors
            "#b4befe",  # Lavender
            "#f5c2e7",  # Mauve
            "#74c7ec",  # Sapphire
            "#94e2d5",  # Teal
            "#fab387",  # Peach
            "#f9e2af",  # Yellow
        ]
    }

    _BASE_FONT_SIZE = 15
    _TABLET_FONT_SIZE = 18
    _TABLET_DIAGONAL_INCH = 9
    _CORNER_RADIUS = 10

    @classmethod
    def apply_theme(cls, app: QApplication) -> None:
        """Apply refined Catppuccin Mocha theme with design system."""
        palette = app.palette()
        colors = cls._CATPPUCCIN_PALETTE

        role_mappings = {
            QPalette.ColorRole.Window: colors["base"],
            QPalette.ColorRole.WindowText: colors["text"],
            QPalette.ColorRole.Base: colors["base"],
            QPalette.ColorRole.AlternateBase: colors["alternate_base"],
            QPalette.ColorRole.ToolTipBase: colors["tooltip_base"],
            QPalette.ColorRole.ToolTipText: colors["text"],
            QPalette.ColorRole.Text: colors["text"],
            QPalette.ColorRole.Button: colors["button"],
            QPalette.ColorRole.ButtonText: colors["button_text"],
            QPalette.ColorRole.BrightText: colors["bright_text"],
            QPalette.ColorRole.Highlight: colors["highlight"],
            QPalette.ColorRole.HighlightedText: colors["highlight_text"],
        }

        for role, color in role_mappings.items():
            palette.setColor(role, QColor(color))

        app.setPalette(palette)
        cls._apply_stylesheet(app)
        cls.apply_adaptive_styles(app)

    @classmethod
    def _apply_stylesheet(cls, app: QApplication) -> None:
        """Apply refined styling with design system consistency."""
        colors = cls._CATPPUCCIN_PALETTE
        radius = cls._CORNER_RADIUS
        stylesheet = f"""
            /* ======== Global Styles ======== */
            QWidget {{
                font-family: "Inter", "Segoe UI", system-ui;
                font-size: {cls._BASE_FONT_SIZE}px;
                color: {colors['text']};
                background: {colors['base']};
            }}

            /* ======== Typography ======== */
            QLabel {{
                font-weight: 450;
                padding: 4px 0;
            }}

            QLabel[important="true"] {{
                font-size: {cls._BASE_FONT_SIZE + 2}px;
                font-weight: 600;
                color: {colors['highlight']};
            }}

            /* ======== Buttons ======== */
            QPushButton {{
                background: {colors['button']};
                color: {colors['button_text']};
                padding: 12px 24px;
                border-radius: {radius}px;
                border: 1px solid {colors['surface1']};
                min-width: 120px;
            }}

            QPushButton:hover {{
                background: {colors['surface1']};
                border-color: {colors['overlay0']};
            }}

            QPushButton:pressed {{
                background: {colors['highlight']};
                color: {colors['highlight_text']};
            }}

            /* ======== Form Elements ======== */
            QLineEdit, QTextEdit, QComboBox {{
                background: {colors['alternate_base']};
                border: 2px solid {colors['surface1']};
                border-radius: {radius - 2}px;
                padding: 12px;
                selection-background-color: {colors['highlight']};
                font: inherit;
            }}

            QLineEdit:focus, QComboBox:focus {{
                border-color: {colors['highlight']};
            }}

            /* ======== Checkboxes & Radio ======== */
            QCheckBox, QRadioButton {{
                spacing: 8px;
                color: {colors['text']};
            }}

            QCheckBox::indicator, QRadioButton::indicator {{
                width: 20px;
                height: 20px;
                border: 2px solid {colors['surface1']};
                border-radius: 4px;
                background: {colors['alternate_base']};
            }}

            QCheckBox::indicator:checked,
            QRadioButton::indicator:checked {{
                background: {colors['highlight']};
                border-color: {colors['highlight']};
            }}

            QCheckBox::indicator:hover,
            QRadioButton::indicator:hover {{
                border-color: {colors['overlay0']};
            }}

            /* ======== Sliders ======== */
            QSlider::groove:horizontal {{
                background: {colors['surface1']};
                height: 6px;
                border-radius: 3px;
            }}

            QSlider::handle:horizontal {{
                background: {colors['highlight']};
                border: 2px solid {colors['surface1']};
                width: 20px;
                height: 20px;
                margin: -8px 0;
                border-radius: 10px;
            }}

            /* ======== Progress Bars ======== */
            QProgressBar {{
                background: {colors['alternate_base']};
                border: 2px solid {colors['surface1']};
                border-radius: {radius - 2}px;
                text-align: center;
                color: {colors['text']};
            }}

            QProgressBar::chunk {{
                background: {colors['highlight']};
                border-radius: {radius - 4}px;
                margin: 2px;
            }}

            /* ======== Tabs ======== */
            QTabWidget {{
                background: transparent;
            }}

            QTabWidget::pane {{
                border: 2px solid {colors['surface1']};
                border-radius: {radius}px;
                margin-top: 8px;
                background: {colors['alternate_base']};
            }}

            QTabBar::tab {{
                background: {colors['button']};
                color: {colors['text']};
                padding: 14px 28px;
                border-top-left-radius: {radius}px;
                border-top-right-radius: {radius}px;
                border: 2px solid transparent;
                margin-right: 6px;
                font-weight: 500;
            }}

            QTabBar::tab:selected {{
                background: {colors['highlight']};
                color: {colors['highlight_text']};
                border-color: {colors['surface1']};
            }}

            /* ======== Data Visualization ======== */
            QChartView {{
                background: {colors['alternate_base']};
                border-radius: {radius}px;
                border: 2px solid {colors['surface1']};
            }}

            /* ======== Lists & Trees ======== */
            QListWidget, QTreeView {{
                background: {colors['alternate_base']};
                border: 2px solid {colors['surface1']};
                border-radius: {radius}px;
                padding: 6px;
                outline: 0;
            }}

            QListWidget::item, QTreeWidget::item {{
                background: {colors['surface1']};
                color: {colors['button_text']};
                padding: 12px;
                border-radius: {radius - 4}px;
                margin: 4px;
            }}

            QListWidget::item:hover,
            QTreeWidget::item:hover {{
                background: {colors['tooltip_base']};
                border-color: {colors['overlay0']};
            }}

            QListWidget::item:selected,
            QTreeWidget::item:selected {{
                background: {colors['highlight']};
                color: {colors['highlight_text']};
            }}

            /* ======== Scrollbars ======== */
            QScrollBar:vertical {{
                background: {colors['base']};
                width: 14px;
                border-radius: {radius}px;
            }}

            QScrollBar::handle:vertical {{
                background: {colors['surface1']};
                min-height: 40px;
                border-radius: {radius}px;
                margin: 4px;
            }}

            QScrollBar:horizontal {{
                background: {colors['base']};
                height: 14px;
                border-radius: {radius}px;
            }}

            QScrollBar::handle:horizontal {{
                background: {colors['surface1']};
                min-width: 40px;
                border-radius: {radius}px;
                margin: 4px;
            }}
        """
        app.setStyleSheet(stylesheet)

    @classmethod
    def apply_chart_theme(cls, chart: QChart) -> None:
        """Apply Catppuccin styling to charts."""
        colors = cls._CATPPUCCIN_PALETTE

        chart.setBackgroundBrush(Qt.GlobalColor.transparent)
        chart.setBackgroundPen(QPen(Qt.GlobalColor.transparent))

        chart.setPlotAreaBackgroundBrush(QColor(colors["alternate_base"]))
        chart.setPlotAreaBackgroundVisible(True)

        title_font = QFont("Inter", cls._BASE_FONT_SIZE + 8)
        title_font.setWeight(QFont.Weight.ExtraBold)
        chart.setTitleFont(title_font)
        chart.setTitleBrush(QColor(colors["highlight"]))

        title_shadow = QGraphicsDropShadowEffect()
        title_shadow.setBlurRadius(15)
        title_shadow.setColor(QColor(colors["base"]))
        title_shadow.setOffset(2, 2)

        axis_color = colors["text"]
        axis_pen = QPen(axis_color)
        axis_pen.setWidth(2)

        grid_pen = QPen(QColor(colors["surface1"]))
        grid_pen.setWidth(1)
        minor_grid_pen = QPen(QColor(colors["base"]))
        minor_grid_pen.setWidth(1)

        for axis in chart.axes():
            if isinstance(axis, (QValueAxis, QBarCategoryAxis)):
                axis.setLabelsColor(QColor(axis_color))
                axis.setTitleBrush(QColor(axis_color))
                axis.setLinePen(axis_pen)
                axis.setGridLinePen(grid_pen)
                axis.setMinorGridLinePen(minor_grid_pen)

        for series in chart.series():
            if isinstance(series, QBarSeries):
                for i, bar_set in enumerate(series.barSets()):
                    color = QColor(
                        colors["graph_colors"][i % len(colors["graph_colors"])]
                    )
                    bar_set.setBrush(color)
                    bar_set.setPen(QPen(color.darker(120), 1))

        legend = chart.legend()
        if legend:
            legend.setLabelColor(QColor(colors["text"]))
            legend.setMarkerShape(QLegend.MarkerShape.MarkerShapeCircle)
            legend.setBackgroundVisible(False)
            legend.setPen(QPen(Qt.GlobalColor.transparent))

    @classmethod
    def apply_adaptive_styles(cls, app: QApplication) -> None:
        """Enhanced adaptive styling for different devices."""
        if cls.is_tablet_device(app):
            tablet_css = f"""
                QWidget {{
                    font-size: {cls._TABLET_FONT_SIZE}px;
                }}

                QPushButton, QTabBar::tab {{
                    padding: 18px 36px;
                    min-width: 160px;
                }}

                QLineEdit, QComboBox {{
                    min-height: 56px;
                    padding: 16px;
                }}

                QListWidget::item {{
                    padding: 20px;
                    min-height: 64px;
                }}

                QChartView {{
                    min-height: 480px;
                }}
            """
            app.setStyleSheet(app.styleSheet() + tablet_css)

    @staticmethod
    def is_tablet_device(app: QApplication) -> bool:
        """Determine if device is tablet-sized."""
        screen = app.primaryScreen()
        diag = (screen.size().width()**2 + screen.size().height()**2)**0.5
        diag_inch = diag / screen.logicalDotsPerInch()
        return diag_inch <= DarkTheme._TABLET_DIAGONAL_INCH


class ValidationError(Exception):
    """Custom exception for validation errors with field context."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None
    ) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


class Logger:
    """Centralized logging component with configurable levels."""

    LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

    def __init__(self) -> None:
        self._level = "INFO"

    def configure(self, level: str) -> None:
        """Set the current logging level."""
        if level in self.LEVELS:
            self._level = level

    def log(
        self,
        level: str,
        module: str,
        message: str,
        *args: Any
    ) -> None:
        """Base logging method with level checking."""
        if self.LEVELS.index(level) < self.LEVELS.index(self._level):
            return

        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        formatted_msg = message % args
        print(f"{timestamp} [{level}] [{module}] {formatted_msg}")

    def debug(
        self,
        module: str,
        message: str,
        *args: Any
    ) -> None:
        """Log debug-level message."""
        self.log("DEBUG", module, message, *args)

    def info(
        self,
        module: str,
        message: str,
        *args: Any
    ) -> None:
        """Log info-level message."""
        self.log("INFO", module, message, *args)

    def warning(
        self,
        module: str,
        message: str,
        *args: Any
    ) -> None:
        """Log warning-level message."""
        self.log("WARNING", module, message, *args)

    def error(
        self,
        module: str,
        message: str,
        *args: Any
    ) -> None:
        """Log error-level message."""
        self.log("ERROR", module, message, *args)
# endregion

# region MODELS


class PollModel(QObject):
    """Model representing a single poll with voting data."""

    data_changed = Signal()

    def __init__(
        self,
        title: str,
        options: List[str],
        logger: Logger
    ) -> None:
        super().__init__()
        self.logger = logger
        self.title = title
        self._options: Dict[str, int] = {opt: 0 for opt in options}
        self.voter_ids: Set[str] = set()
        self._cached_total: int = 0

    def add_vote(self, user_id: str, option: str) -> bool:
        """Add a vote from a user to the specified option."""
        if user_id in self.voter_ids:
            self.logger.warning(
                "PollModel",
                "Duplicate vote attempt from %s",
                user_id
            )
            return False

        if option not in self._options:
            self.logger.error(
                "PollModel",
                "Invalid option '%s' in poll '%s'",
                option,
                self.title
            )
            return False

        self._options[option] += 1
        self.voter_ids.add(user_id)
        self._cached_total += 1
        self.data_changed.emit()
        return True

    def reset_votes(self) -> None:
        """Reset all voting data for this poll."""
        self._options = {k: 0 for k in self._options}
        self.voter_ids.clear()
        self._cached_total = 0
        self.data_changed.emit()

    def update_options(self, new_options: List[str]) -> None:
        """Replace existing options with new set of choices."""
        self._options = {opt: 0 for opt in new_options}
        self.voter_ids.clear()
        self._cached_total = 0
        self.data_changed.emit()

    @property
    def total_votes(self) -> int:
        """Current total number of votes in this poll."""
        return self._cached_total

    @property
    def options(self) -> Dict[str, int]:
        """Copy of voting options with current vote counts."""
        return self._options.copy()

    @staticmethod
    def validate_user_id(user_id: str) -> None:
        """Validate user ID format against allowed patterns."""
        if len(user_id) < 3:
            raise ValidationError(
                _("ID must contain at least 3 characters"),
                "user_id"
            )

        patterns = (
            r'^\d+$',
            r'^[\w\.-]+@[\w\.-]+\.\w+$',
            r'^[\da-f]{8}-([\da-f]{4}-){3}[\da-f]{12}$'
        )

        if not any(re.match(p, user_id) for p in patterns):
            raise ValidationError(_("Invalid ID format"), "user_id")


class AppModel(QObject):
    """Main application model managing polls and admin state."""

    polls_changed = Signal()
    admin_status_changed = Signal(bool)

    def __init__(self, logger: Logger) -> None:
        super().__init__()
        self.logger = logger
        self._polls: List[PollModel] = []
        self._admin_logged_in: bool = False
        self._admin_credentials: Tuple[str, str] = ("admin", "admin")

    def create_poll(self, title: str, options: List[str]) -> PollModel:
        """Create and store a new poll with validation."""
        existing_titles = [p.title for p in self._polls]
        self._validate_poll_title(title, existing_titles)
        self._validate_options(options)

        poll = PollModel(title, options, self.logger)
        self._polls.append(poll)
        self.logger.info(
            "AppModel",
            "Created poll '%s' with options: %s",
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
        """Modify an existing poll with validation."""
        existing_titles = [
            p.title for p in self._polls if p != self._polls[index]
        ]
        self._validate_poll_title(new_title, existing_titles)
        self._validate_options(new_options)

        poll = self._polls[index]
        old_title = poll.title
        poll.title = new_title
        poll.update_options(new_options)
        self.logger.info(
            "AppModel",
            "Updated poll '%s' to '%s' with options: %s",
            old_title,
            new_title,
            new_options
        )
        self.polls_changed.emit()

    def reset_poll(self, index: int) -> None:
        """Reset voting data for a specific poll."""
        self._polls[index].reset_votes()
        self.polls_changed.emit()

    def authenticate_admin(
        self,
        username: str,
        password: str
    ) -> bool:
        """Verify admin credentials and update state."""
        success = (
            username == self._admin_credentials[0] and
            password == self._admin_credentials[1]
        )
        self._admin_logged_in = success
        self.admin_status_changed.emit(success)
        return success

    @property
    def polls(self) -> List[PollModel]:
        """Copy of current polls list."""
        return self._polls.copy()

    @property
    def is_admin(self) -> bool:
        """Current admin authentication status."""
        return self._admin_logged_in

    @staticmethod
    def _validate_poll_title(
        title: str,
        existing_titles: List[str]
    ) -> None:
        """Ensure poll title meets requirements."""
        if not title.strip():
            raise ValidationError(
                _("Poll title cannot be empty"),
                "title"
            )
        if title in existing_titles:
            raise ValidationError(
                _("Poll title must be unique"),
                "title"
            )

    @staticmethod
    def _validate_options(options: List[str]) -> None:
        """Validate option list integrity."""
        if len(options) < 2:
            raise ValidationError(
                _("Minimum 2 options required"),
                "options"
            )
        if any(not opt.strip() for opt in options):
            raise ValidationError(
                _("Options cannot be empty"),
                "options"
            )
        if len(options) != len(set(options)):
            raise ValidationError(
                _("Duplicate options detected"),
                "options"
            )
# endregion

# region VIEWS


class BaseView(QWidget):
    """Base widget for all views with common configuration."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Dialog)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)


class MainView(QWidget):
    """Main application window with poll management controls."""

    poll_selected = Signal(int)
    admin_login_requested = Signal()
    admin_logout_requested = Signal()
    add_poll_requested = Signal()
    edit_poll_requested = Signal(int)

    def __init__(self, logger: Logger) -> None:
        super().__init__()
        self.logger = logger
        self._init_ui()
        self.setMinimumSize(400, 500)

    def _init_ui(self) -> None:
        """Initialize and arrange UI components."""
        self.setWindowTitle(_("Voting System"))
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
        """Configure poll selection list widget."""
        self.poll_list = QListWidget()
        self.poll_list.itemDoubleClicked.connect(
            lambda: self.poll_selected.emit(self.poll_list.currentRow())
        )

        self.poll_list_group = QGroupBox(_("Available Polls"))
        layout = QVBoxLayout()
        layout.addWidget(self.poll_list)
        self.poll_list_group.setLayout(layout)

    def _create_user_group(self) -> None:
        """Build user voting interface components."""
        self.user_group = QGroupBox(_("Voting"))
        layout = QVBoxLayout()

        self.vote_btn = QPushButton(_("Vote"))
        self.vote_btn.clicked.connect(
            lambda: self.poll_selected.emit(self.poll_list.currentRow())
        )

        layout.addWidget(self.vote_btn)
        self.user_group.setLayout(layout)

    def _create_admin_group(self) -> None:
        """Construct admin control elements."""
        self.admin_group = QGroupBox(_("Admin Panel"))
        layout = QVBoxLayout()

        self.admin_login_btn = QPushButton(_("Login"))
        self.admin_login_btn.clicked.connect(
            self.admin_login_requested.emit
        )

        self.add_poll_btn = QPushButton(_("Create Poll"))
        self.add_poll_btn.clicked.connect(
            self.add_poll_requested.emit
        )
        self.add_poll_btn.setVisible(False)

        self.edit_poll_btn = QPushButton(_("Edit Poll"))
        self.edit_poll_btn.clicked.connect(
            self._on_edit_poll
        )
        self.edit_poll_btn.setVisible(False)

        self.results_btn = QPushButton(_("View Results"))
        self.results_btn.clicked.connect(
            self._show_results
        )
        self.results_btn.setVisible(False)

        self.logout_btn = QPushButton(_("Logout"))
        self.logout_btn.clicked.connect(
            self._confirm_logout
        )
        self.logout_btn.setVisible(False)

        layout.addWidget(self.admin_login_btn)
        layout.addWidget(self.add_poll_btn)
        layout.addWidget(self.edit_poll_btn)
        layout.addWidget(self.results_btn)
        layout.addWidget(self.logout_btn)
        self.admin_group.setLayout(layout)

    def _on_edit_poll(self) -> None:
        """Handle edit poll button click."""
        index = self.poll_list.currentRow()
        if index >= 0:
            self.edit_poll_requested.emit(index)

    def _show_results(self) -> None:
        """Handle results viewing request."""
        index = self.poll_list.currentRow()
        if index >= 0:
            self.poll_selected.emit(index)

    def _confirm_logout(self) -> None:
        """Confirm admin logout through dialog."""
        confirm = QMessageBox.question(
            self,
            _("Confirmation"),
            _("Are you sure you want to logout from admin mode?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.admin_logout_requested.emit()

    def update_polls(self, polls: List[PollModel]) -> None:
        """Synchronize displayed polls with model data."""
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

    def update_admin_status(self, is_admin: bool) -> None:
        """Update UI elements based on admin state."""
        status = _("Logged in as admin") if is_admin else _("User mode")
        self.status_label.setText(status)

        self.user_group.setVisible(not is_admin)
        self.vote_btn.setVisible(not is_admin)
        self.add_poll_btn.setVisible(is_admin)
        self.edit_poll_btn.setVisible(is_admin)
        self.admin_login_btn.setVisible(not is_admin)
        self.logout_btn.setVisible(is_admin)


class AdminAuthView(BaseView):
    """Authentication dialog for admin access."""

    login_attempt = Signal(str, str)

    def __init__(self, logger: Logger) -> None:
        super().__init__()
        self.logger = logger
        self._init_ui()

    def _init_ui(self) -> None:
        """Configure authentication form elements."""
        layout = QFormLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText(_("Admin username"))

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText(_("Password"))

        self.login_btn = QPushButton(_("Login"))

        layout.addRow(_("Username:"), self.username_input)
        layout.addRow(_("Password:"), self.password_input)
        layout.addWidget(self.login_btn)

        self.login_btn.clicked.connect(
            lambda: self.login_attempt.emit(
                self.username_input.text(),
                self.password_input.text()
            )
        )
        self.setLayout(layout)


class PollCreatorView(BaseView):
    """Dialog for creating new polls."""

    create_poll = Signal(str, list)

    def __init__(self, logger: Logger) -> None:
        super().__init__()
        self.logger = logger
        self._init_ui()

    def _init_ui(self) -> None:
        """Set up poll creation form elements."""
        layout = QFormLayout()

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText(_("Poll title"))

        self.options_input = QLineEdit()
        self.options_input.setPlaceholderText(
            _("Option1, Option2, ...")
        )

        self.create_btn = QPushButton(_("Create"))

        layout.addRow(_("Title:"), self.title_input)
        layout.addRow(_("Options (comma-separated):"), self.options_input)
        layout.addWidget(self.create_btn)

        self.create_btn.clicked.connect(self._on_create)
        self.setLayout(layout)

    def _on_create(self) -> None:
        """Handle poll creation form submission."""
        title = self.title_input.text().strip()
        raw_options = self.options_input.text()

        try:
            options = [
                opt.strip()
                for opt in shlex.split(raw_options, posix=False)
                if opt.strip()
            ]
        except ValueError as e:
            QMessageBox.warning(
                self,
                _("Error"),
                _("Invalid option formatting: %s") % str(e)
            )
            return

        self.create_poll.emit(title, options)


class PollEditorView(BaseView):
    """Dialog for modifying existing polls."""

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
        """Configure poll editing form elements."""
        layout = QFormLayout()

        self.title_input = QLineEdit(self.poll.title)
        self.title_input.setPlaceholderText(_("New poll title"))

        self.options_input = QLineEdit(
            ", ".join(self.poll.options.keys())
        )
        self.options_input.setPlaceholderText(
            _("Updated options, comma-separated")
        )

        self.update_btn = QPushButton(_("Update"))

        layout.addRow(_("Title:"), self.title_input)
        layout.addRow(_("Options:"), self.options_input)
        layout.addWidget(self.update_btn)

        self.update_btn.clicked.connect(self._on_update)
        self.setLayout(layout)

    def _on_update(self) -> None:
        """Handle poll update form submission."""
        title = self.title_input.text().strip()
        raw_options = self.options_input.text()

        try:
            options = [
                opt.strip()
                for opt in shlex.split(raw_options, posix=False)
                if opt.strip()
            ]
        except ValueError as e:
            QMessageBox.warning(
                self,
                _("Error"),
                _("Invalid option formatting: %s") % str(e)
            )
            return

        self.update_poll.emit(title, options)


class VotingView(BaseView):
    """Voting interface with results visualization."""

    vote_submitted = Signal(str, str)

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
        self.setWindowTitle(_("Voting Results - {}").format(poll.title))

    def _init_ui(self) -> None:
        """Initialize and arrange voting interface components."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        if self.readonly:
            self._create_chart(main_layout)
            self._create_options_list(main_layout)
        else:
            self._create_form(main_layout)
            self._create_options_list(main_layout)
            self._create_buttons(main_layout)

    def _create_form(self, layout: QVBoxLayout) -> None:
        """Build user input form for voting."""
        form = QFormLayout()
        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText(_("Your unique identifier"))
        form.addRow(QLabel(_("User ID:")), self.user_id_input)
        layout.addLayout(form)

    def _create_chart(self, layout: QVBoxLayout) -> None:
        """Configure visualization chart for results."""
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.chart_view, 3)

    def _create_options_list(self, layout: QVBoxLayout) -> None:
        """Set up options selection/results list."""
        self.options_list = QListWidget()

        if self.readonly:
            self.options_list.setSelectionMode(
                QListWidget.SelectionMode.NoSelection
            )
        else:
            self.options_list.setSelectionMode(
                QListWidget.SelectionMode.SingleSelection
            )

        layout.addWidget(self.options_list)

    def _create_buttons(self, layout: QVBoxLayout) -> None:
        """Create action buttons for voting interface."""
        btn_layout = QHBoxLayout()
        self.vote_btn = QPushButton(_("Vote"))
        self.vote_btn.clicked.connect(self._on_vote)
        btn_layout.addWidget(self.vote_btn)
        layout.addLayout(btn_layout)

    def showEvent(self, event: QShowEvent) -> None:
        """Handle window display event to initialize chart."""
        if self.readonly and not self._chart_initialized:
            self._init_chart()
            self._chart_initialized = True
        self._update_view()
        super().showEvent(event)

    def _init_chart(self) -> None:
        """Initialize chart properties and axes."""
        self.chart = QChart()
        self.chart.setBackgroundRoundness(10)
        DarkTheme.apply_chart_theme(self.chart)
        self.chart_view.setChart(self.chart)

    def _update_chart(self) -> None:
        """Update chart data based on current voting results."""
        self.chart.removeAllSeries()

        for axis in self.chart.axes():
            self.chart.removeAxis(axis)

        bar_set = QBarSet(_("Votes"))
        values = list(self.poll.options.values())
        bar_set.append(values)

        series = QBarSeries()
        series.setBarWidth(0.6)
        series.append(bar_set)
        self.chart.addSeries(series)

        max_value = max(values) if values else 0
        upper = max(max_value, 1)
        categories = [
            f"{opt}\n({votes})"
            for opt, votes in self.poll.options.items()
        ]

        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        axis_x.setTitleText(_("Voting Options"))

        axis_y = QValueAxis()
        axis_y.setTitleText(_("Vote Count"))
        axis_y.setRange(0, upper)
        axis_y.setTickInterval(1)

        self.chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)

        DarkTheme.apply_chart_theme(self.chart)

    def _update_options(self) -> None:
        """Refresh options list display with current data."""
        self.options_list.clear()
        max_votes = max(self.poll.options.values(), default=0)

        for option, votes in self.poll.options.items():
            if self.readonly:
                item_text = f"{option}: {votes} {_('votes')}"
            else:
                item_text = option

            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, option)

            if self.readonly and votes == max_votes:
                item.setBackground(Qt.GlobalColor.green)
                item.setForeground(Qt.GlobalColor.white)

            self.options_list.addItem(item)

    def _update_view(self) -> None:
        """Complete refresh of all visual components."""
        try:
            self.setUpdatesEnabled(False)
            if self.readonly:
                self._update_chart()
            self._update_options()
            if self.readonly:
                self.chart.setTitle(
                    _("Voting Results - {}").format(self.poll.title)
                )
        finally:
            self.setUpdatesEnabled(True)

    def _on_vote(self) -> None:
        """Handle vote submission attempt."""
        self.vote_btn.setEnabled(False)
        try:
            user_id = self.user_id_input.text().strip()
            option = self._get_selected_option()

            PollModel.validate_user_id(user_id)
            if not option:
                raise ValidationError(_("Select an option"), "option")

            self.vote_submitted.emit(user_id, option)
        except ValidationError as e:
            QMessageBox.warning(self, _("Error"), e.message)
        finally:
            self.vote_btn.setEnabled(True)

    def _get_selected_option(self) -> Optional[str]:
        """Retrieve currently selected voting option."""
        if item := self.options_list.currentItem():
            return item.text().split(":")[0].strip()
        return None
# endregion

# region CONTROLLERS


class BaseController(QObject):
    """Base controller with signal management capabilities."""

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
        """Track and manage signal connections."""
        connection = signal.connect(slot)
        self._connections.append(connection)

    def cleanup(self) -> None:
        """Disconnect all managed signal connections."""
        for connection in self._connections:
            QObject.disconnect(connection)
        self._connections.clear()


class Container(containers.DeclarativeContainer):
    """Dependency injection container for component management."""

    logger = providers.Singleton(Logger)
    model = providers.Factory(AppModel, logger=logger)
    main_view = providers.Factory(MainView, logger=logger)
    auth_view = providers.Factory(AdminAuthView, logger=logger)
    poll_creator_view = providers.Factory(PollCreatorView, logger=logger)
    voting_view = providers.Factory(VotingView, logger=logger)
    poll_editor_view = providers.Factory(PollEditorView, logger=logger)


class MainController(BaseController):
    """Central controller managing application workflow."""

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
        """Connect view signals to controller methods."""
        connections = [
            (self.view.poll_selected, self._show_voting),
            (self.view.add_poll_requested, self._show_poll_creator),
            (self.view.admin_login_requested, self._show_admin_login),
            (self.view.admin_logout_requested, self._logout_admin),
            (self.view.edit_poll_requested, self._show_poll_editor),
            (self.model.polls_changed, self._update_polls_display),
            (self.model.admin_status_changed, self._update_admin_ui),
        ]

        for signal, handler in connections:
            self.manage_connection(signal, handler)

    def _show_voting(self, index: int) -> None:
        """Display voting interface for selected poll."""
        if not 0 <= index < len(self.model.polls):
            QMessageBox.warning(self.view, _("Error"), _("Invalid poll"))
            return

        poll = self.model.polls[index]
        view = self._voting_view_factory(poll, self.model.is_admin)
        controller = VotingController(poll, view, self.model.logger)
        controller.setParent(view)
        view.show()

    def _show_admin_login(self) -> None:
        """Display admin authentication dialog."""
        dialog = self._auth_view_factory()
        controller = AdminAuthController(self.model, dialog)
        controller.setParent(dialog)
        dialog.show()

    def _show_poll_creator(self) -> None:
        """Display poll creation dialog."""
        dialog = self._poll_creator_factory()
        controller = PollCreatorController(self.model, dialog)
        controller.setParent(dialog)
        dialog.show()

    def _show_poll_editor(self, index: int) -> None:
        """Display poll editing dialog."""
        if not 0 <= index < len(self.model.polls):
            QMessageBox.warning(self.view, _("Error"), _("Invalid poll"))
            return

        poll = self.model.polls[index]
        dialog = self._poll_editor_factory(poll)
        controller = PollEditorController(self.model, dialog, index)
        controller.setParent(dialog)
        dialog.show()

    def _logout_admin(self) -> None:
        """Handle admin logout process."""
        self.model.authenticate_admin("", "")

    def _update_polls_display(self) -> None:
        """Refresh displayed poll list from model."""
        self.view.update_polls(self.model.polls)

    def _update_admin_ui(self, is_admin: bool) -> None:
        """Update UI elements based on admin state changes."""
        self.view.update_admin_status(is_admin)


class AdminAuthController(BaseController):
    """Controller handling admin authentication flow."""

    def __init__(
        self,
        model: AppModel,
        view: AdminAuthView
    ) -> None:
        super().__init__()
        self.model = model
        self.view = view
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect authentication signals."""
        self.manage_connection(
            self.view.login_attempt,
            self._authenticate
        )

    def _authenticate(
        self,
        username: str,
        password: str
    ) -> None:
        """Process admin login attempt."""
        if self.model.authenticate_admin(username, password):
            self.view.close()
            QMessageBox.information(
                self.view,
                _("Success"),
                _("Logged in as administrator")
            )
        else:
            QMessageBox.warning(
                self.view,
                _("Error"),
                _("Invalid administrator credentials")
            )


class PollCreatorController(BaseController):
    """Controller managing poll creation process."""

    def __init__(
        self,
        model: AppModel,
        view: PollCreatorView
    ) -> None:
        super().__init__()
        self.model = model
        self.view = view
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect poll creation signals."""
        self.manage_connection(
            self.view.create_poll,
            self._create_poll
        )

    def _create_poll(
        self,
        title: str,
        options: List[str]
    ) -> None:
        """Handle new poll submission."""
        try:
            self.model.create_poll(title, options)
            self.view.close()
            QMessageBox.information(
                self.view,
                _("Success"),
                _("Poll created successfully")
            )
        except ValidationError as e:
            QMessageBox.warning(self.view, _("Error"), e.message)


class PollEditorController(BaseController):
    """Controller managing poll modification process."""

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
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect poll update signals."""
        self.manage_connection(
            self.view.update_poll,
            self._update_poll
        )

    def _update_poll(
        self,
        title: str,
        options: List[str]
    ) -> None:
        """Handle poll update submission."""
        try:
            self.model.edit_poll(self.poll_index, title, options)
            self.view.close()
            QMessageBox.information(
                self.view,
                _("Success"),
                _("Poll updated successfully")
            )
        except ValidationError as e:
            QMessageBox.warning(self.view, _("Error"), e.message)


class VotingController(BaseController):
    """Controller managing voting process and results display."""

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
        self._connect_signals()
        QTimer.singleShot(0, self._update_view)

    def _connect_signals(self) -> None:
        """Connect voting-related signals."""
        self.manage_connection(
            self.view.vote_submitted,
            self._process_vote
        )
        self.manage_connection(
            self.poll.data_changed,
            self._update_view
        )

    def _update_view(self) -> None:
        """Refresh voting interface display."""
        self.view._update_view()

    def _process_vote(
        self,
        user_id: str,
        option: str
    ) -> None:
        """Process submitted vote and update model."""
        try:
            if not self.poll.add_vote(user_id, option):
                QMessageBox.warning(
                    self.view,
                    _("Error"),
                    _("You already voted in this poll")
                )
                return

            self.view.user_id_input.clear()
            self.view.options_list.clearSelection()
            QMessageBox.information(
                self.view,
                _("Success"),
                _("Vote submitted successfully")
            )
        except ValidationError as e:
            QMessageBox.warning(self.view, _("Error"), e.message)
# endregion


# region MAIN
if __name__ == "__main__":
    app = QApplication(sys.argv)
    DarkTheme.apply_theme(app)
    DarkTheme.apply_adaptive_styles(app)
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
# endregion
