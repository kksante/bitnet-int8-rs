"""A small held-out English corpus spanning genres (encyclopedic, narrative,
dialogue, instructional, code, news). It is deliberately diverse but small; it is
a *mini-corpus* for a controlled quantization study, not WikiText/C4. Swapping in
a real held-out set is a drop-in change (see wikitext_loader.py / ppl.corpus_nll).
"""

CORPUS = [
    # encyclopedic / scientific
    "The mitochondria is the powerhouse of the cell, generating most of the chemical energy needed to power biochemical reactions.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy absorbed from sunlight by chlorophyll.",
    "The speed of light in a vacuum is approximately 299,792 kilometres per second, a fundamental constant of the universe.",
    "Plate tectonics describes the slow movement of large sections of the Earth's lithosphere over the underlying mantle.",
    # historical / news
    "In 1969, Apollo 11 landed the first humans on the Moon, and Neil Armstrong became the first person to step onto its surface.",
    "The treaty was signed after months of negotiation, and both nations agreed to reduce tariffs on agricultural goods.",
    "Central banks raised interest rates again on Thursday, citing persistent inflation and a resilient labour market.",
    # narrative / dialogue
    "She opened the old wooden door slowly, half expecting the hinges to scream, but the room beyond was silent and full of dust.",
    "\"I never expected to see you here,\" he said, lowering his voice as the train pulled away from the crowded platform.",
    "The mountain path narrowed as they climbed, and by dusk the village lights below had shrunk to a scatter of distant sparks.",
    # instructional
    "To make a simple loaf of bread you need flour, water, salt, and yeast; mix them, let the dough rise, then bake until golden.",
    "Before changing a tyre, park on level ground, apply the handbrake, and loosen the wheel nuts before raising the car with the jack.",
    # code / technical
    "A binary search repeatedly divides a sorted array in half, comparing the target to the middle element until it is found.",
    "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
    "In Python, a list comprehension such as [x * x for x in range(10)] builds a new list by applying an expression to each item.",
    # general knowledge
    "William Shakespeare wrote approximately thirty-seven plays and over one hundred and fifty sonnets during his lifetime.",
]
