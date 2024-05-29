from transformers import pipeline
from utils import setup_logger

logger = setup_logger(__name__)


class Summarizer:
    def __init__(self, model_name: str = "google/pegasus-xsum"):
        self.summarizer = pipeline("summarization", model=model_name)
        logger.info(f"Summarizer: {model_name} loaded correctly")

    def summarize(self, text: str, max_length: int = 50,
                  min_length: int = 15) -> str:
        logger.info("Summarizing...")
        summary = self.summarizer(
            text, max_length=max_length, min_length=min_length, do_sample=False
        )
        return summary[0]['summary_text']


if __name__ == "__main__":
    summarizer = Summarizer()

    history = (
        "Stellar Blade is an action-adventure"
        "video game set in a post-apocalyptic"
        "[1] Combat focuses on grasping the"
        "attack patterns of enemies"
        "by parrying and evading in combat."
        "Eve also has a Burst Gauge filled"
        "through successive parries and combos"
    )

    summary = summarizer.summarize(history)
    print(summary)
