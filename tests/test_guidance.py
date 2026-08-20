from src.ecosort.guidance import DEFAULT_GUIDANCE, disposal_guidance


def test_known_class_has_guidance():
    assert "local" in disposal_guidance("plastic").lower()


def test_unknown_class_uses_default():
    assert disposal_guidance("mystery") == DEFAULT_GUIDANCE


def test_class_name_is_normalized():
    assert disposal_guidance("  GLASS ") == disposal_guidance("glass")
