from transformers import pipeline
from utils import setup_logger

logger = setup_logger(__name__)

class Summarizer:
    def __init__(self, model_name: str = "google/pegasus-xsum"):
        self.summarizer = pipeline("summarization", model=model_name)
        logger.info(f"Summarizer model: {model_name} loaded correctly")

    def summarize(self, text: str, max_length: int = 50, min_length: int = 15) -> str:
        logger.info("Summarizing...")
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

if __name__ == "__main__":
    summarizer = Summarizer()

    history = (
       "Stellar Blade is an action-adventure video game set in a post-apocalyptic Earth and played from a third-person perspective.[1] Combat focuses on grasping the attack patterns of enemies and countering with precise timing. Eve's Beta Gauge is filled by parrying and evading in combat. The Beta Gauge is spent to use skills such as piercing super armor and enemy combo interruption. Eve also has a Burst Gauge filled through successive parries and combos, which can be used to activate buffs and powerful attacks. The game utilizes the PlayStation 5 DualSense controller's haptic feedback to deliver feedback on enemy attacks and weapon accuracy. Exploration features wall scaling and swinging on ropes for environmental traversal, with things to be found including extra costumes. The overworld has NPCs who can provide Eve with side quests, with the option of either taking them on or ignoring them.[2]"
    )

    summary = summarizer.summarize(history)
    print(summary)
