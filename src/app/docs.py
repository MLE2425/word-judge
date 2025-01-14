from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import MarkdownViewer, Footer

with open("README.md") as f:
    EXAMPLE_MARKDOWN = f.read()


class Docs(Screen):
    BINDINGS = [
        ("escape", "app.switch_mode('home')", "return to the home screen"),
    ]

    def compose(self) -> ComposeResult:
        yield MarkdownViewer(EXAMPLE_MARKDOWN, show_table_of_contents=True)
        yield Footer()
