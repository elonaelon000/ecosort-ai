"""Generic disposal guidance for broad material classes.

Local recycling rules vary, so this module intentionally avoids claiming that
an item is accepted by a specific municipality.
"""

DEFAULT_GUIDANCE = (
    "Check local recycling and waste rules before disposal. Material composition, "
    "contamination, and local facilities can change what is accepted."
)

GUIDANCE = {
    "cardboard": "Keep it clean and dry. Flatten it if your local recycling program accepts cardboard.",
    "glass": "If accepted locally, rinse containers and separate caps or lids when required.",
    "metal": "Rinse food or drink residue and check whether your local program accepts this type of metal.",
    "paper": "Keep paper clean and dry. Greasy or heavily contaminated paper may not be recyclable.",
    "plastic": "Check the plastic type and local rules. Not every plastic item is accepted by every program.",
    "trash": "This item may not fit the supported recyclable categories. Check local disposal guidance.",
}


def disposal_guidance(class_name: str) -> str:
    return GUIDANCE.get(class_name.strip().lower(), DEFAULT_GUIDANCE)
