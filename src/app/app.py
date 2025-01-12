from textual.app import App
from textual.binding import Binding
from src.app.home import Home
from src.app.main import MainLoader, Main
from src.app.docs import Docs


class WordJugdeApp(App):
    CSS = """
        .column {
            align: center top;
            &>*{ max-width: 100; }
        }
        Screen .-maximized {
            margin: 1 2;
            max-width: 100%;
            &.column { margin: 1 2; padding: 1 2; }
            &.column > * {
                max-width: 100%;
            }
        }
    """
    MODES = {
        "home": Home,
        "mainloader": MainLoader,
        "main": Main,
        "docs": Docs,
    }
    DEFAULT_MODE = "home"

    BINDINGS = [
        Binding("ctrl+q", "quit", "quit", tooltip="quit the application"),
        Binding(
            "ctrl+h", "app.switch_mode('home')", "home", tooltip="go to the home screen"
        ),
        Binding(
            "ctrl+j",
            "app.switch_mode('mainloader')",
            "judge",
            tooltip="go to the application",
        ),
        Binding(
            "ctrl+d",
            "app.switch_mode('docs')",
            "docs",
            tooltip="go to the documentation",
        ),
    ]

    def on_mount(self) -> None:
        self.theme = "tokyo-night"


def app():
    app = WordJugdeApp()
    app.run()


if __name__ == "__main__":
    app()
