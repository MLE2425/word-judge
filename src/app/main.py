from textual import on
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Vertical
from textual.widgets import TextArea, Footer, Input, LoadingIndicator

from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from src.data.nlp import tokenize
from src.app.utils import load_model, load_tokenizer

TEXT = """\
Any text input here will be analyzed by wordjudge to determine whether it is AI generated or not.
"""


class MainLoader(Screen):
    async def on_mount(self) -> None:
        self.app.call_later(self.on_loaded)

    def on_loaded(self) -> None:
        model, tfidf_vectorizer = load_model()
        tokenizer, lemmatizer, stopwords, punctuation = load_tokenizer()
        self.app.push_screen(
            Main(
                model=model,
                vectorizer=tfidf_vectorizer,
                tokenizer=tokenizer,
                lemmatizer=lemmatizer,
                stopwords=stopwords,
                punctuation=punctuation,
            )
        )

    def compose(self) -> ComposeResult:
        yield LoadingIndicator("Loading...")


class Main(Screen):
    CSS = """
        Main {
            hatch: cross red;
            TextArea {
                border: $text-primary double;
                margin: 1;
            }
            Input {
                border: $text-secondary double;
                margin: 1;
            }
        }
    """

    def __init__(
        self,
        model: BaseEstimator,
        vectorizer: TfidfVectorizer,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        lemmatizer: WordNetLemmatizer,
        stopwords: set[str],
        punctuation: set[str],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.tfidf_vectorizer = vectorizer
        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer
        self.stopwords = stopwords
        self.punctuation = punctuation

    BINDINGS = [
        (
            "escape",
            "app.switch_mode('home')",
            "return to the home screen",
        ),
        (
            "ctrl+j",
            "submit()",
            "submit the text",
        ),
    ]

    def compose(self) -> ComposeResult:
        txtarea = TextArea(TEXT, language="python", id="input-box")
        txtarea.border_title = "introduce the text to analyze"
        result_box = Input("not yet analyzed", id="result-box", disabled=True)
        result_box.border_title = "result"
        yield Vertical(txtarea, result_box)
        yield Footer()

    @on(TextArea.Changed)
    def on_text_area_changed(self) -> None:
        """Handle the event when the text in the input box changes."""
        result_box = self.query_one("#result-box", Input)
        result_box.value = "Not yet analyzed"

    async def action_submit(self) -> None:
        """Handle the event when the user submits the text by pressing Enter."""
        input_text = self.query_one("#input-box", TextArea).text

        result_box = self.query_one("#result-box", Input)

        result_box.value = "Analyzing..."
        result = self.analyze_text(input_text)

        result_box.value = result

    def analyze_text(self, text: str) -> str:
        """Analyze the input text and return a result."""
        tokens = (
            tokenize(
                text,
                True,
                True,
                self.tokenizer,
                self.stopwords,
                self.punctuation,
                self.lemmatizer,
            )
            .__str__()
            .replace("'", "")
            .replace("[", "")
            .replace("]", "")
            .replace(",", "")
            .replace("CLS ", "")
            .replace("SEP", "")
            .replace("  ", " ")
        )

        text_tfidf = self.tfidf_vectorizer.transform([tokens])

        predictions = self.model.predict(text_tfidf)

        match predictions[0]:
            case 0:
                return "Human generated"
            case 1:
                return "AI generated"
            case _:
                return "Unknown"
