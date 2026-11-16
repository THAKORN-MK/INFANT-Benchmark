from pathlib import Path


def check_output_location(dataset_root: Path, output_root: Path) -> None:
    dataset, output = dataset_root.resolve(), output_root.resolve()
    if output == dataset or output.is_relative_to(dataset):
        raise ValueError('Output must not be inside the dataset directory')
