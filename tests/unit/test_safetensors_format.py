from __future__ import annotations

from pathlib import Path

from inkling_ampere.quantization.safetensors import (
    build_layout,
    initialize_file,
    read_layout,
)


def test_streaming_layout_round_trips_without_tensor_dependencies(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    layout = build_layout(
        [
            ("z.weight", "BF16", (2, 4)),
            ("a.weight_packed", "I32", (2, 1)),
        ],
        metadata={"format": "pt", "plan": "test"},
    )

    initialize_file(path, layout)
    specs = layout.tensor_map()
    with path.open("r+b") as handle:
        handle.seek(layout.data_offset + specs["a.weight_packed"].data_start)
        handle.write(bytes(range(8)))
        handle.seek(layout.data_offset + specs["z.weight"].data_start)
        handle.write(bytes(range(16)))

    observed = read_layout(path)

    assert observed.metadata == {"format": "pt", "plan": "test"}
    assert observed.file_bytes == path.stat().st_size
    assert observed.tensor_map() == specs
