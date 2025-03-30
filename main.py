# -*- coding: utf-8 -*-

# region IMPORTS
import sys
import re
from datetime import datetime
from typing import List, Dict, Set, Optional, Callable
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
)
from PySide6.QtGui import QPainter, QShowEvent
from PySide6.QtCharts import (
    QChart,
    QChartView,
    QBarSet,
    QBarSeries,
    QBarCategoryAxis,
    QValueAxis,
)
from dependency_injector import containers, providers
# endregion

# region UTILITIES


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.field = field
        self.message = message


class Logger:
    """Centralized logging component."""

    LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

    def __init__(self) -> None:
        self._level = "INFO"

    def configure(self, level: str) -> None:
        """Set logging level."""
        if level in self.LEVELS:
            self._level = level

    def log(self, level: str, module: str, message: str) -> None:
        """Base logging method."""
        if self.LEVELS.index(level) < self.LEVELS.index(self._level):
            return

        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        print(f"{timestamp} [{level}] [{module}] {message}")

    def debug(self, module: str, message: str) -> None:
        """Log debug message."""
        self.log("DEBUG", module, message)

    def info(self, module: str, message: str) -> None:
        """Log info message."""
        self.log("INFO", module, message)

    def warning(self, module: str, message: str) -> None:
        """Log warning message."""
        self.log("WARNING", module, message)

    def error(self, module: str, message: str) -> None:
        """Log error message."""
        self.log("ERROR", module, message)
# endregion

# region MODELS


class PollModel(QObject):
    """Model representing a single poll."""

    data_changed = Signal()

    def __init__(self, title: str, options: List[str], logger: Logger):
        super().__init__()
        self.logger = logger
        self.title = title
        self._options = {opt: 0 for opt in options}
        self.voter_ids: Set[str] = set()
        self._cached_total = 0

    def add_vote(self, user_id: str, option: str) -> bool:
        """Add a vote to the poll."""
        if user_id in self.voter_ids:
            self.logger.warning(
                "PollModel",
                f"Duplicate vote attempt from {user_id}"
            )
            return False

        if option not in self._options:
            self.logger.error(
                "PollModel",
                f"Invalid option '{option}' in poll '{self.title}'"
            )
            return False

        self._options[option] += 1
        self.voter_ids.add(user_id)
        self._cached_total += 1
        self.data_changed.emit()
        return True

    def reset_votes(self) -> None:
        """Reset all votes for this poll."""
        self._options = {opt: 0 for opt in self._options.keys()}
        self.voter_ids.clear()
        self._cached_total = 0
        self.data_changed.emit()

    def update_options(self, new_options: List[str]) -> None:
        """Update poll options."""
        self._options = {opt: 0 for opt in new_options}
        self.voter_ids.clear()
        self._cached_total = 0
        self.data_changed.emit()

    @property
    def total_votes(self) -> int:
        """Get total number of votes."""
        return self._cached_total

    @property
    def options(self) -> Dict[str, int]:
        """Get copy of voting options."""
        return self._options.copy()

    @staticmethod
    def validate_user_id(user_id: str) -> None:
        """Validate user ID format."""
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
    """Main application model."""

    polls_changed = Signal()
    admin_status_changed = Signal(bool)

    def __init__(self, logger: Logger):
        super().__init__()
        self.logger = logger
        self._polls: List[PollModel] = []
        self._admin_logged_in = False
        self._admin_credentials = ("admin", "admin")

    def create_poll(self, title: str, options: List[str]) -> PollModel:
        """Create new poll."""
        existing_titles = [p.title for p in self._polls]
        self._validate_poll_title(title, existing_titles)
        self._validate_options(options)

        poll = PollModel(title, options, self.logger)
        self._polls.append(poll)
        self.polls_changed.emit()
        return poll

    def edit_poll(
        self,
        index: int,
        new_title: str,
        new_options: List[str]
    ) -> None:
        """Edit existing poll."""
        existing_titles = [
            p.title for p in self._polls if p != self._polls[index]
        ]
        self._validate_poll_title(new_title, existing_titles)
        self._validate_options(new_options)

        poll = self._polls[index]
        poll.title = new_title
        poll.update_options(new_options)
        self.polls_changed.emit()

    def reset_poll(self, index: int) -> None:
        """Reset poll votes."""
        self._polls[index].reset_votes()
        self.polls_changed.emit()

    def authenticate_admin(self, username: str, password: str) -> bool:
        """Authenticate admin user."""
        success = (
            username == self._admin_credentials[0] and
            password == self._admin_credentials[1]
        )
        self._admin_logged_in = success
        self.admin_status_changed.emit(success)
        return success

    @property
    def polls(self) -> List[PollModel]:
        """Get copy of polls list."""
        return self._polls.copy()

    @property
    def is_admin(self) -> bool:
        """Check admin status."""
        return self._admin_logged_in

    @staticmethod
    def _validate_poll_title(title: str, existing_titles: List[str]) -> None:
        """Validate poll title."""
        if not title.strip():
            raise ValidationError(_("Poll title cannot be empty"), "title")
        if title in existing_titles:
            raise ValidationError(_("Poll title must be unique"), "title")

    @staticmethod
    def _validate_options(options: List[str]) -> None:
        """Validate poll options."""
        if len(options) < 2:
            raise ValidationError(_("Minimum 2 options required"), "options")
        if any(not opt.strip() for opt in options):
            raise ValidationError(_("Options cannot be empty"), "options")
        if len(options) != len(set(options)):
            raise ValidationError(_("Duplicate options detected"), "options")
# endregion

# region VIEWS


class BaseView(QWidget):
    """Base class for all views."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Dialog)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)


class MainView(QWidget):
    """Main application view."""

    poll_selected = Signal(int)
    admin_login_requested = Signal()
    admin_logout_requested = Signal()
    add_poll_requested = Signal()
    edit_poll_requested = Signal(int)

    def __init__(self, logger: Logger):
        super().__init__()
        self.logger = logger
        self._init_ui()
        self.setMinimumSize(400, 500)

    def _init_ui(self) -> None:
        """Initialize UI components."""
        self.setWindowTitle(_("Voting System"))
        main_layout = QVBoxLayout()

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("background: #4b4b4b; padding: 5px;")

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
        """Create polls list widget."""
        self.poll_list = QListWidget()
        self.poll_list.itemDoubleClicked.connect(
            lambda: self.poll_selected.emit(self.poll_list.currentRow())
        )

        self.poll_list_group = QGroupBox(_("Available Polls"))
        layout = QVBoxLayout()
        layout.addWidget(self.poll_list)
        self.poll_list_group.setLayout(layout)

    def _create_user_group(self) -> None:
        """Create user voting section."""
        self.user_group = QGroupBox(_("Voting"))
        layout = QVBoxLayout()

        self.vote_btn = QPushButton(_("Vote"))
        self.vote_btn.clicked.connect(
            lambda: self.poll_selected.emit(self.poll_list.currentRow())
        )

        layout.addWidget(self.vote_btn)
        self.user_group.setLayout(layout)

    def _create_admin_group(self) -> None:
        """Create admin controls section."""
        self.admin_group = QGroupBox(_("Admin Panel"))
        layout = QVBoxLayout()

        self.admin_login_btn = QPushButton(_("Login"))
        self.admin_login_btn.clicked.connect(self.admin_login_requested.emit)

        self.add_poll_btn = QPushButton(_("Create Poll"))
        self.add_poll_btn.clicked.connect(self.add_poll_requested.emit)
        self.add_poll_btn.setVisible(False)

        self.edit_poll_btn = QPushButton(_("Edit Poll"))
        self.edit_poll_btn.clicked.connect(self._on_edit_poll)
        self.edit_poll_btn.setVisible(False)

        self.results_btn = QPushButton(_("View Results"))
        self.results_btn.clicked.connect(self._show_results)
        self.results_btn.setVisible(False)

        self.logout_btn = QPushButton(_("Logout"))
        self.logout_btn.clicked.connect(self._confirm_logout)
        self.logout_btn.setVisible(False)

        layout.addWidget(self.admin_login_btn)
        layout.addWidget(self.add_poll_btn)
        layout.addWidget(self.edit_poll_btn)
        layout.addWidget(self.results_btn)
        layout.addWidget(self.logout_btn)
        self.admin_group.setLayout(layout)

    def _on_edit_poll(self) -> None:
        """Handle edit poll request."""
        index = self.poll_list.currentRow()
        if index >= 0:
            self.edit_poll_requested.emit(index)

    def _show_results(self) -> None:
        """Handle show results request."""
        index = self.poll_list.currentRow()
        if index >= 0:
            self.poll_selected.emit(index)

    def _confirm_logout(self) -> None:
        """Confirm admin logout."""
        confirm = QMessageBox.question(
            self,
            _("Confirmation"),
            _("Are you sure you want to logout from admin mode?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.admin_logout_requested.emit()

    def update_polls(self, polls: List[PollModel]) -> None:
        """Update displayed polls list."""
        current_titles = {
            self.poll_list.item(i).text()
            for i in range(self.poll_list.count())
        }
        new_titles = {poll.title for poll in polls}

        for title in current_titles - new_titles:
            items = self.poll_list.findItems(
                title, Qt.MatchFlag.MatchExactly
            )
            for item in items:
                self.poll_list.takeItem(self.poll_list.row(item))

        for poll in polls:
            if poll.title not in current_titles:
                item = QListWidgetItem(poll.title)
                self.poll_list.addItem(item)

    def update_admin_status(self, is_admin: bool) -> None:
        """Update UI based on admin status."""
        status = _("Logged in as admin") if is_admin else _("User mode")
        self.status_label.setText(status)
        self.user_group.setVisible(not is_admin)
        self.vote_btn.setVisible(not is_admin)
        self.add_poll_btn.setVisible(is_admin)
        self.edit_poll_btn.setVisible(is_admin)
        self.admin_login_btn.setVisible(not is_admin)
        self.logout_btn.setVisible(is_admin)


class AdminAuthView(BaseView):
    """Admin authentication view."""

    login_attempt = Signal(str, str)

    def __init__(self, logger: Logger):
        super().__init__()
        self.logger = logger
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI components."""
        layout = QFormLayout()
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
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
    """Poll creation view."""

    create_poll = Signal(str, list)

    def __init__(self, logger: Logger):
        super().__init__()
        self.logger = logger
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI components."""
        layout = QFormLayout()
        self.title_input = QLineEdit()
        self.options_input = QLineEdit()
        self.create_btn = QPushButton(_("Create"))

        layout.addRow(_("Title:"), self.title_input)
        layout.addRow(_("Options (comma-separated):"), self.options_input)
        layout.addWidget(self.create_btn)

        self.create_btn.clicked.connect(self._on_create)
        self.setLayout(layout)

    def _on_create(self) -> None:
        """Handle poll creation."""
        title = self.title_input.text().strip()
        options = [opt.strip() for opt in self.options_input.text().split(",")]
        self.create_poll.emit(title, options)


class PollEditorView(BaseView):
    """Poll editing view."""

    update_poll = Signal(str, list)

    def __init__(self, poll: PollModel, logger: Logger):
        super().__init__()
        self.logger = logger
        self.poll = poll
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI components."""
        layout = QFormLayout()
        self.title_input = QLineEdit(self.poll.title)
        self.options_input = QLineEdit(", ".join(self.poll.options.keys()))
        self.update_btn = QPushButton(_("Update"))

        layout.addRow(_("Title:"), self.title_input)
        layout.addRow(_("Options (comma-separated):"), self.options_input)
        layout.addWidget(self.update_btn)

        self.update_btn.clicked.connect(self._on_update)
        self.setLayout(layout)

    def _on_update(self) -> None:
        """Handle poll update."""
        title = self.title_input.text().strip()
        options = [opt.strip() for opt in self.options_input.text().split(",")]
        self.update_poll.emit(title, options)


class VotingView(BaseView):
    """Voting/results view."""

    vote_submitted = Signal(str, str)

    def __init__(self, poll: PollModel, readonly: bool, logger: Logger):
        super().__init__()
        self.poll = poll
        self.logger = logger
        self.readonly = readonly
        self._chart_initialized = False
        self._init_ui()
        self._init_chart()
        self.setMinimumSize(800, 600)
        self.setWindowTitle(_("Voting Results - {}").format(poll.title))

    def _init_ui(self) -> None:
        """Initialize UI components."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self._create_form(main_layout)
        self._create_chart(main_layout)
        self._create_options_list(main_layout)
        self._create_buttons(main_layout)
        self._apply_styles()

    def _create_form(self, main_layout: QVBoxLayout) -> None:
        """Create user input form."""
        form = QFormLayout()
        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText(_("Your ID"))
        form.addRow(QLabel(_("User ID:")), self.user_id_input)
        main_layout.addLayout(form)

    def _create_chart(self, main_layout: QVBoxLayout) -> None:
        """Create chart widget."""
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        main_layout.addWidget(self.chart_view, 3)

    def _create_options_list(self, main_layout: QVBoxLayout) -> None:
        """Create options list widget."""
        self.options_list = QListWidget()
        self.options_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection
        )
        main_layout.addWidget(self.options_list)

    def _create_buttons(self, main_layout: QVBoxLayout) -> None:
        """Create action buttons."""
        btn_layout = QHBoxLayout()
        self.vote_btn = QPushButton(_("Vote"))
        self.vote_btn.clicked.connect(self._on_vote)
        btn_layout.addWidget(self.vote_btn)
        main_layout.addLayout(btn_layout)

    def _apply_styles(self) -> None:
        """Apply CSS styling."""
        self.setStyleSheet("""
            QChartView {
                background-color: #f5f5f5;
                border-radius: 5px;
            }
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #cccccc;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)

    def showEvent(self, event: QShowEvent) -> None:
        """Handle view show event."""
        if not self._chart_initialized:
            self._init_chart()
            self._chart_initialized = True
        self._update_view()
        super().showEvent(event)

    def _init_chart(self) -> None:
        """Initialize chart properties."""
        self.chart = QChart()
        self.chart.setBackgroundRoundness(10)
        self.chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        self.chart_view.setChart(self.chart)

    def _update_chart(self) -> None:
        """Update chart data."""
        self.chart.removeAllSeries()

        bar_set = QBarSet(_("Votes"))
        bar_set.setColor("#2196F3")
        values = list(self.poll.options.values())
        bar_set.append(values)

        series = QBarSeries()
        series.setBarWidth(0.6)
        series.append(bar_set)
        self.chart.addSeries(series)

        max_value = max(values) if values else 1
        categories = [
            f"{opt}\n({votes})"
            for opt, votes in self.poll.options.items()
        ]

        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        axis_x.setTitleText(_("Voting Options"))

        axis_y = QValueAxis()
        axis_y.setTitleText(_("Vote Count"))
        axis_y.setRange(0, max_value * 1.1)

        self.chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)

    def _update_options(self) -> None:
        """Update options list display."""
        self.options_list.clear()
        max_votes = max(self.poll.options.values(), default=0)

        for option, votes in self.poll.options.items():
            item = QListWidgetItem(f"{option}: {votes} {_('votes')}")
            item.setData(Qt.ItemDataRole.UserRole, option)
            if votes == max_votes:
                item.setBackground(Qt.GlobalColor.green)
                item.setForeground(Qt.GlobalColor.white)
            self.options_list.addItem(item)

    def _update_view(self) -> None:
        """Update complete view state."""
        try:
            self.setUpdatesEnabled(False)
            self._update_chart()
            self._update_options()
            self.chart.setTitle(
                _("Voting Results - {}").format(self.poll.title)
            )
        finally:
            self.setUpdatesEnabled(True)

    def _on_vote(self) -> None:
        """Handle vote submission."""
        self.vote_btn.setEnabled(False)
        try:
            user_id = self.user_id_input.text().strip()
            option = self._get_selected_option()

            PollModel.validate_user_id(user_id)
            if not option:
                raise ValidationError(_("Select an option"), "option")

            self.vote_submitted.emit(user_id, option)
            QMessageBox.information(self, _("Success"), _("Vote recorded!"))
            self.close()

        except ValidationError as e:
            self._highlight_error(e.field)
            QMessageBox.warning(self, _("Error"), e.message)
        finally:
            self.vote_btn.setEnabled(True)

    def _get_selected_option(self) -> Optional[str]:
        """Get currently selected option."""
        if item := self.options_list.currentItem():
            return item.text().split(":")[0].strip()
        return None

    def _highlight_error(self, field: Optional[str]) -> None:
        """Visually highlight invalid fields."""
        if field is None:
            return

        widgets = {
            "user_id": self.user_id_input,
            "option": self.options_list
        }

        if widget := widgets.get(field):
            def clear_highlight() -> None:
                widget.setStyleSheet("")

            widget.setStyleSheet("border: 2px solid red;")
            QTimer.singleShot(2000, clear_highlight)
# endregion

# region CONTROLLERS


class BaseController(QObject):
    """Base controller with proper signal management."""

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._connections: List[QMetaObject.Connection] = []

    def manage_connection(
        self,
        signal: SignalInstance,
        slot: Callable
    ) -> None:
        """Properly manage signal connections with type safety."""
        connection = signal.connect(slot)
        self._connections.append(connection)

    def cleanup(self) -> None:
        """Clean up all managed connections."""
        for connection in self._connections:
            self.disconnect(connection)
        self._connections.clear()


class Container(containers.DeclarativeContainer):
    """Dependency injection container."""

    logger = providers.Singleton(Logger)
    model = providers.Factory(AppModel, logger=logger)
    main_view = providers.Factory(MainView, logger=logger)
    auth_view = providers.Factory(AdminAuthView, logger=logger)
    poll_creator_view = providers.Factory(PollCreatorView, logger=logger)
    voting_view = providers.Factory(VotingView, logger=logger)
    poll_editor_view = providers.Factory(PollEditorView, logger=logger)


class MainController(BaseController):
    """Main application controller."""

    def __init__(
        self,
        model: AppModel,
        view: MainView,
        auth_view_factory: Callable[[], AdminAuthView],
        poll_creator_factory: Callable[[], PollCreatorView],
        poll_editor_factory: Callable[[PollModel], PollEditorView],
        voting_view_factory: Callable[[PollModel, bool], VotingView]
    ):
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
        self.manage_connection(
            self.view.poll_selected,
            self._show_voting
        )
        self.manage_connection(
            self.view.add_poll_requested,
            self._show_poll_creator
        )
        self.manage_connection(
            self.view.admin_login_requested,
            self._show_admin_login
        )
        self.manage_connection(
            self.view.admin_logout_requested,
            self._logout_admin
        )
        self.manage_connection(
            self.view.edit_poll_requested,
            self._show_poll_editor
        )
        self.manage_connection(
            self.model.polls_changed,
            self._update_polls_display
        )
        self.manage_connection(
            self.model.admin_status_changed,
            self._update_admin_ui
        )

    def _show_voting(self, index: int) -> None:
        """Show voting view for selected poll."""
        if not 0 <= index < len(self.model.polls):
            QMessageBox.warning(self.view, _("Error"), _("Invalid poll"))
            return

        poll = self.model.polls[index]
        view = self._voting_view_factory(poll, self.model.is_admin)
        controller = VotingController(poll, view, self.model.logger)
        controller.setParent(view)
        view.show()

    def _show_admin_login(self) -> None:
        """Show admin authentication view."""
        dialog = self._auth_view_factory()
        controller = AdminAuthController(self.model, dialog)
        controller.setParent(dialog)
        dialog.show()

    def _show_poll_creator(self) -> None:
        """Show poll creation view."""
        dialog = self._poll_creator_factory()
        controller = PollCreatorController(self.model, dialog)
        controller.setParent(dialog)
        dialog.show()

    def _show_poll_editor(self, index: int) -> None:
        """Show poll editor view."""
        if not 0 <= index < len(self.model.polls):
            QMessageBox.warning(self.view, _("Error"), _("Invalid poll"))
            return

        poll = self.model.polls[index]
        dialog = self._poll_editor_factory(poll)
        controller = PollEditorController(self.model, dialog, index)
        controller.setParent(dialog)
        dialog.show()

    def _logout_admin(self) -> None:
        """Handle admin logout."""
        self.model.authenticate_admin("", "")

    def _update_polls_display(self) -> None:
        """Update displayed polls list."""
        self.view.update_polls(self.model.polls)

    def _update_admin_ui(self, is_admin: bool) -> None:
        """Update UI for admin status changes."""
        self.view.update_admin_status(is_admin)


class AdminAuthController(BaseController):
    """Admin authentication controller."""

    def __init__(self, model: AppModel, view: AdminAuthView):
        super().__init__()
        self.model = model
        self.view = view
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect authentication signals."""
        self.manage_connection(self.view.login_attempt, self._authenticate)

    def _authenticate(self, username: str, password: str) -> None:
        """Handle authentication attempt."""
        if self.model.authenticate_admin(username, password):
            self.view.close()
            QMessageBox.information(self.view, _("Success"), _("Logged in"))
        else:
            QMessageBox.warning(self.view, _("Error"),
                                _("Invalid credentials"))


class PollCreatorController(BaseController):
    """Poll creation controller."""

    def __init__(self, model: AppModel, view: PollCreatorView):
        super().__init__()
        self.model = model
        self.view = view
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect poll creation signals."""
        self.manage_connection(self.view.create_poll, self._create_poll)

    def _create_poll(self, title: str, options: List[str]) -> None:
        """Handle poll creation."""
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
    """Poll editing controller."""

    def __init__(self, model: AppModel, view: PollEditorView, poll_index: int):
        super().__init__()
        self.model = model
        self.view = view
        self.poll_index = poll_index
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect poll update signals."""
        self.manage_connection(self.view.update_poll, self._update_poll)

    def _update_poll(self, title: str, options: List[str]) -> None:
        """Handle poll update."""
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
    """Voting process controller."""

    def __init__(self, poll: PollModel, view: VotingView, logger: Logger):
        super().__init__(view)
        self.poll = poll
        self.view = view
        self.logger = logger
        self._connect_signals()
        self._update_view()

    def _connect_signals(self) -> None:
        """Connect voting signals."""
        self.manage_connection(self.view.vote_submitted, self._process_vote)
        self.manage_connection(self.poll.data_changed, self._update_view)

    def _update_view(self) -> None:
        """Update voting view."""
        self.view._update_view()

    def _process_vote(self, user_id: str, option: str) -> None:
        """Process vote submission."""
        try:
            if not self.poll.add_vote(user_id, option):
                QMessageBox.warning(
                    self.view,
                    _("Error"),
                    _("You already voted in this poll")
                )
                return

            QMessageBox.information(
                self.view,
                _("Success"),
                _("Vote recorded successfully")
            )
            self.view.close()
        except ValidationError as e:
            QMessageBox.warning(self.view, _("Error"), e.message)
        except Exception as e:
            self.logger.error("VotingController", str(e))
            QMessageBox.warning(
                self.view,
                _("Error"),
                _("Failed to process vote")
            )
# endregion


# region MAIN
if __name__ == "__main__":
    app = QApplication(sys.argv)
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
