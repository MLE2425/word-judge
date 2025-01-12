from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Label, Footer, Markdown


ART = """\
 ____ ____ ____ ____ _________ ____ ____ ____ ____ ____
||w |||o |||r |||d |||       |||j |||u |||d |||g |||e ||
||__|||__|||__|||__|||_______|||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/_______\|/__\|/__\|/__\|/__\|/__\|
"""

INTRO = """\
# welcome to wordjudge

### about

wordjudge is a simple application that classifies text to detect content generated by large language models (llms).

### controls

- press `q` to quit the application.
- press `h` to return to the home screen.
- press `m` to navigate to the main application.
- press `d` to access the documentation.

---

### credits

developed by [antonio rodriguez ruiz](https://github.com/AntonioRodriguezRuiz)
and [adrián romero flores](https://github.com/adrrf).

check out the repository: [wordjudge](https://github.com/MLE2425/word-judge).

created for the machine learning engineering course,
master's degree in software engineering - cloud, data, and it management,
university of seville - 2024/2025.
"""


class Home(Screen):
    CSS = """
        Home {
            align-horizontal: center;
            Markdown {
                padding-right: 32;
                padding-left: 32;
                text-align: center;
                align-horizontal: center;
                background: $background;
            }
            Label {
                width: 100%;
                padding: 1;
                margin: 1;
                text-align: center;
                align-horizontal: center;
                color: $text-primary;
                background: $primary-muted;
                border: $text-primary;
            }
        }
    """

    def compose(self) -> ComposeResult:
        yield Label(ART)
        yield Markdown(INTRO)
        yield Footer()