# -*- coding: utf-8 -*-


import sys
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QApplication,
    QListWidgetItem
)
from PySide6.QtCore import (
    Signal,
    QObject
)


# region Model


class VotingModel(QObject):
    data_changed = Signal()

    def __init__(self, options):
        super().__init__()
        self._options = {option: 0 for option in options}

    @property
    def options(self):
        return self._options.copy()

    def add_vote(self, option):
        if option in self._options:
            self._options[option] += 1
            self.data_changed.emit()
            return True
        return False
# endregion


# region View


class VotingView(QWidget):
    vote_requested = Signal(str)

    def __init__(self, title):
        super().__init__()
        self._title = title
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Voting App')
        layout = QVBoxLayout()

        self.title_label = QLabel(self._title)
        self.options_list = QListWidget()
        self.vote_button = QPushButton('Vote')

        layout.addWidget(self.title_label)
        layout.addWidget(self.options_list)
        layout.addWidget(self.vote_button)

        self.setLayout(layout)
        self.vote_button.clicked.connect(self._on_vote_clicked)

    def update_options(self, options):
        self.options_list.clear()
        for option, votes in options.items():
            item = QListWidgetItem(f"{option}: {votes} votes")
            self.options_list.addItem(item)

    def _on_vote_clicked(self):
        selected = self.options_list.currentItem()
        if selected:
            option = selected.text().split(':')[0].strip()
            self.vote_requested.emit(option)
# endregion


# region Controller


class VotingController:
    def __init__(self, model, view):
        self._model = model
        self._view = view

        self._connect_signals()
        self._initial_update()

    def _connect_signals(self):
        self._view.vote_requested.connect(self.handle_vote)
        self._model.data_changed.connect(self._update_view)

    def _initial_update(self):
        self._update_view()

    def handle_vote(self, option):
        if not self._model.add_vote(option):
            print(f"Error: Invalid option '{option}'")

    def _update_view(self):
        self._view.update_options(self._model.options)
# endregion


# region Main


def main():
    app = QApplication(sys.argv)

    initial_options = ['Option 1', 'Option 2', 'Option 3']
    model = VotingModel(initial_options)
    view = VotingView("Simple Voting System")
    VotingController(model, view)

    view.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
# endregion
