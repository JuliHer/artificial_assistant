from src.processing.generate_wakeword_dataset import WakeWordDatasetGenerator
from src.processing.augmentation import WakeWordAugmenter
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent



ARTIFICIAL_NOISES_DIR = project_root / "Data" / "audio" / "raw" / "noise"
NATURAL_NOISES_DIR = project_root / "Data" / "audio" / "processed" / "noise"
WAKE_DIR = project_root / "Data" / "audio" / "raw" / "kora"

WAKE_OUTPUT_DIR = project_root / "Data" / "audio" / "processed" / "wake_word"
OUTPUT_DIR = project_root / "Data" / "audio" / "processed" / "dataset"

augmenter = WakeWordAugmenter()
generator = WakeWordDatasetGenerator(augmenter, WAKE_OUTPUT_DIR, ARTIFICIAL_NOISES_DIR, NATURAL_NOISES_DIR, OUTPUT_DIR)

generator.generate_augmented_wake(WAKE_DIR, WAKE_OUTPUT_DIR)

generator.generate_wake_w_noises()
generator.generate_noises_w_wake()
generator.pass_noises_to_dataset()

generator.save_annotations()