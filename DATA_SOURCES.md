# Data Sources and Attribution

## TrashNet

EcoSort AI uses the resized TrashNet dataset for its first version.

- **Dataset:** TrashNet
- **Creators:** Gary Thung and Mindy Yang
- **Original project:** Stanford CS 229 final project
- **Original repository:** https://github.com/garythung/trashnet
- **Current dataset mirror:** https://huggingface.co/datasets/garythung/trashnet
- **License shown by the source repository / dataset mirror:** MIT
- **Resized archive:** `dataset-resized.zip`
- **Archive SHA-256:** `c060e8abfe5d6de0578ca15be1ed8ad0794a865d333c3473d53d1d9ad6e38b8c`

The original repository reports 2,527 resized images:

| Class | Images |
|---|---:|
| glass | 501 |
| paper | 594 |
| cardboard | 403 |
| plastic | 482 |
| metal | 410 |
| trash | 137 |
| **Total** | **2,527** |

The original dataset was photographed largely against a white posterboard under sunlight and/or indoor lighting. This is an important limitation: a model trained on it may perform worse on cluttered real-world scenes, unusual lighting, damaged items, or mixed-material objects.

## Redistribution choice

EcoSort AI does not include the downloaded images in its Git repository. The `scripts/download_trashnet.py` helper downloads them directly from the source mirror and verifies the archive hash.

This keeps the project repository small and keeps attribution explicit.
