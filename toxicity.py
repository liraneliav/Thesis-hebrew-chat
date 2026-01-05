import numpy as np
from transformers import pipeline
from typing import Sequence, Tuple
import matplotlib.pyplot as plt

# Create the pipeline ONCE (module-level) so it's not reloaded each call.
# device: -1 = CPU, 0 = first GPU (if you KNOW you have CUDA). CPU is safest default.
_TOXICITY_PIPE = pipeline(
    task="text-classification",
    model="textdetox/glot500-toxicity-classifier",
    return_all_scores=True,
    device=-1,
    truncation=True
)

_LABEL_MAP = {
    "LABEL_0": "non-toxic",
    "LABEL_1": "toxic"
}

def measuring_toxicity(message: str) -> dict[str, float]:
    """
    Run the glot500 toxicity classifier and return a dict like:
    {"non-toxic": 0.83, "toxic": 0.17}
    """
    outputs = _TOXICITY_PIPE(str(message))

    records = outputs[0] if isinstance(outputs, list) and len(outputs) and isinstance(outputs[0], list) else outputs

    results: dict[str, float] = {}
    for rec in records:
        # normalize label
        lbl = rec["label"]
        if isinstance(lbl, np.ndarray):
            lbl = lbl.item()
        lbl = _LABEL_MAP.get(lbl, str(lbl))

        # normalize score
        sc = rec["score"]
        if isinstance(sc, np.ndarray):
            sc = sc.item()
        sc = float(sc)

        results[lbl] = sc

    results.setdefault("non-toxic", 0.0)
    results.setdefault("toxic", 0.0)
    return results

def plot_toxicity(
    turns: Sequence[int],
    user_toxicity_scores: Sequence[float],
    persona_toxicity_scores: Sequence[float],
    configuration: str = ""
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot toxicity scores across dialogue turns.

    Returns (fig, ax) so you can save or embed the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(turns, user_toxicity_scores, marker="o", label="User Toxicity")
    ax.plot(turns, persona_toxicity_scores, marker="x", label="Persona Toxicity")
    ax.set_xlabel("Response Number")
    ax.set_ylabel("Toxicity Score (Probability)")
    title = f"Toxicity Over Dialogue Turns {configuration}".strip()
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    txt = "I totally disagree with you, but let's discuss calmly you stupid."
    print(measuring_toxicity(txt))

    turns = [1, 2, 3, 4, 5, 6]
    user = [0.10, 0.22, 0.18, 0.31, 0.27, 0.35]
    persona = [0.05, 0.08, 0.07, 0.09, 0.11, 0.10]
    fig, ax = plot_toxicity(turns, user, persona, configuration="[Chat 1]")
    plt.show()