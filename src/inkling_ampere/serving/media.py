"""Fail-closed admission for self-contained Responses media inputs."""

from __future__ import annotations

import base64
import binascii
import io
import math
import struct
import wave
import zlib
from dataclasses import dataclass
from typing import Any

from inkling_ampere.serving.profile import MultimodalSettings


class MediaAdmissionError(ValueError):
    """Raised before enqueue when media violates the reviewed contract."""


@dataclass(frozen=True)
class MediaItem:
    modality: str
    format: str
    mime_type: str
    transport_characters: int
    file_bytes: int
    decoded_bytes: int
    processor_tokens: int
    width: int | None = None
    height: int | None = None
    processed_width: int | None = None
    processed_height: int | None = None
    frames: int | None = None
    duration_seconds: float | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    sample_width_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in {
                "modality": self.modality,
                "format": self.format,
                "mime_type": self.mime_type,
                "transport_characters": self.transport_characters,
                "file_bytes": self.file_bytes,
                "decoded_bytes": self.decoded_bytes,
                "processor_tokens": self.processor_tokens,
                "width": self.width,
                "height": self.height,
                "processed_width": self.processed_width,
                "processed_height": self.processed_height,
                "frames": self.frames,
                "duration_seconds": self.duration_seconds,
                "sample_rate_hz": self.sample_rate_hz,
                "channels": self.channels,
                "sample_width_bytes": self.sample_width_bytes,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class MediaAdmissionReport:
    items: tuple[MediaItem, ...]

    @property
    def image_count(self) -> int:
        return sum(item.modality == "image" for item in self.items)

    @property
    def audio_count(self) -> int:
        return sum(item.modality == "audio" for item in self.items)

    @property
    def total_decoded_bytes(self) -> int:
        return sum(item.decoded_bytes for item in self.items)

    @property
    def total_processor_tokens(self) -> int:
        return sum(item.processor_tokens for item in self.items)

    def to_dict(self) -> dict[str, object]:
        return {
            "image_count": self.image_count,
            "audio_count": self.audio_count,
            "total_items": len(self.items),
            "total_decoded_bytes": self.total_decoded_bytes,
            "total_processor_tokens": self.total_processor_tokens,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class _PngHeader:
    width: int
    height: int
    bit_depth: int
    color_type: int
    compression: int
    filtering: int
    interlace: int


def _decode_base64(value: object, field: str, *, maximum_characters: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise MediaAdmissionError(f"{field} must be a non-empty base64 string")
    if len(value) > maximum_characters:
        raise MediaAdmissionError(f"{field} exceeds the base64 character limit")
    try:
        ascii_value = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise MediaAdmissionError(f"{field} must contain ASCII base64") from exc
    try:
        return base64.b64decode(ascii_value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaAdmissionError(f"{field} is malformed base64") from exc


def _decode_data_url(value: object, *, maximum_characters: int) -> tuple[str, str, bytes]:
    if not isinstance(value, str) or not value.startswith("data:"):
        raise MediaAdmissionError("input_image.image_url must be a base64 data URL")
    header, separator, encoded = value.partition(",")
    if not separator or not header.endswith(";base64"):
        raise MediaAdmissionError("input_image.image_url must use ;base64 encoding")
    mime_type = header.removeprefix("data:").removesuffix(";base64").lower()
    if not mime_type or ";" in mime_type:
        raise MediaAdmissionError("input_image.image_url has an invalid MIME type")
    payload = _decode_base64(
        encoded,
        "input_image.image_url",
        maximum_characters=maximum_characters,
    )
    return mime_type, encoded, payload


def _png_chunks(payload: bytes) -> tuple[_PngHeader, bytes]:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise MediaAdmissionError("image MIME type does not match the PNG signature")
    offset = 8
    header: _PngHeader | None = None
    compressed = bytearray()
    saw_palette = False
    saw_idat = False
    idat_closed = False
    saw_end = False
    while offset < len(payload):
        if len(payload) - offset < 12:
            raise MediaAdmissionError("PNG contains a truncated chunk")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        if (
            len(chunk_type) != 4
            or any(
                value not in range(ord("A"), ord("Z") + 1)
                and value not in range(ord("a"), ord("z") + 1)
                for value in chunk_type
            )
            or chr(chunk_type[2]).islower()
        ):
            raise MediaAdmissionError("PNG chunk type is invalid")
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(payload):
            raise MediaAdmissionError("PNG chunk length exceeds the file")
        chunk_data = payload[data_start:data_end]
        expected_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        observed_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if observed_crc != expected_crc:
            raise MediaAdmissionError("PNG chunk CRC is invalid")
        if chunk_type == b"IHDR":
            if header is not None or length != 13 or offset != 8:
                raise MediaAdmissionError("PNG IHDR is invalid")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            header = _PngHeader(
                width=width,
                height=height,
                bit_depth=bit_depth,
                color_type=color_type,
                compression=compression,
                filtering=filtering,
                interlace=interlace,
            )
        elif chunk_type == b"IDAT":
            if header is None or saw_end or idat_closed:
                raise MediaAdmissionError("PNG IDAT is out of order")
            saw_idat = True
            compressed.extend(chunk_data)
        elif chunk_type == b"PLTE":
            if (
                header is None
                or saw_palette
                or saw_idat
                or length == 0
                or length > 768
                or length % 3
            ):
                raise MediaAdmissionError("PNG PLTE is invalid")
            saw_palette = True
        elif chunk_type == b"IEND":
            if length != 0 or header is None or not compressed:
                raise MediaAdmissionError("PNG IEND is invalid")
            saw_end = True
            offset = crc_end
            break
        else:
            if chunk_type[0] & 32 == 0:
                raise MediaAdmissionError("PNG contains an unsupported critical chunk")
            if saw_idat:
                idat_closed = True
        offset = crc_end
    if header is None or not saw_end or offset != len(payload):
        raise MediaAdmissionError("PNG is missing a terminal IEND chunk")
    return header, bytes(compressed)


def _inspect_png(
    payload: bytes,
    *,
    maximum_width: int,
    maximum_height: int,
    maximum_pixels: int,
    maximum_decoded_bytes: int,
) -> tuple[int, int, int]:
    header, compressed = _png_chunks(payload)
    width = header.width
    height = header.height
    bit_depth = header.bit_depth
    color_type = header.color_type
    if width <= 0 or height <= 0:
        raise MediaAdmissionError("PNG dimensions must be positive")
    if width > maximum_width or height > maximum_height:
        raise MediaAdmissionError("image resolution exceeds the profile limit")
    if width * height > maximum_pixels:
        raise MediaAdmissionError("image pixel count exceeds the profile limit")
    if bit_depth != 8 or color_type not in {2, 6}:
        raise MediaAdmissionError("PNG must be 8-bit RGB or RGBA")
    if header.compression != 0 or header.filtering != 0 or header.interlace != 0:
        raise MediaAdmissionError("PNG must use standard compression, filtering, and no interlace")
    channels = 3 if color_type == 2 else 4
    decoded_bytes = width * height * channels
    if decoded_bytes > maximum_decoded_bytes:
        raise MediaAdmissionError("image decoded size exceeds the profile limit")
    row_bytes = width * channels
    expected_scanline_bytes = height * (row_bytes + 1)
    decompressor = zlib.decompressobj()
    try:
        scanlines = decompressor.decompress(compressed, expected_scanline_bytes + 1)
        if len(scanlines) > expected_scanline_bytes or decompressor.unconsumed_tail:
            raise MediaAdmissionError("PNG decoded scanline size exceeds its header")
        remaining = expected_scanline_bytes + 1 - len(scanlines)
        scanlines += decompressor.flush(remaining)
    except zlib.error as exc:
        raise MediaAdmissionError("PNG IDAT stream is malformed") from exc
    if (
        len(scanlines) != expected_scanline_bytes
        or decompressor.unconsumed_tail
        or decompressor.unused_data
        or not decompressor.eof
    ):
        raise MediaAdmissionError("PNG decoded scanline size is invalid")
    stride = row_bytes + 1
    if any(scanlines[offset] > 4 for offset in range(0, len(scanlines), stride)):
        raise MediaAdmissionError("PNG contains an invalid scanline filter")
    return width, height, decoded_bytes


def _scaled_image_dimensions(
    width: int,
    height: int,
    *,
    factor: float,
    maximum_upscaled_long_edge: int,
) -> tuple[int, int]:
    long_edge = max(width, height)
    target_long_edge = min(
        float(long_edge) * factor,
        float(max(maximum_upscaled_long_edge, long_edge)),
    )
    ratio = target_long_edge / long_edge
    return (
        max(1, math.floor(width * ratio + 0.5)),
        max(1, math.floor(height * ratio + 0.5)),
    )


def inspect_image_part(part: dict[str, Any], settings: MultimodalSettings) -> MediaItem:
    image = settings.image
    processor = settings.processor
    if not settings.enabled or not image.enabled:
        raise MediaAdmissionError("image input is not enabled by this serving profile")
    if processor is None:
        raise MediaAdmissionError("image processor configuration is missing")
    if part.get("detail", "auto") != "auto":
        raise MediaAdmissionError("input_image.detail must be auto")
    mime_type, encoded, payload = _decode_data_url(
        part.get("image_url"),
        maximum_characters=image.maximum_base64_characters,
    )
    if mime_type not in image.mime_types:
        raise MediaAdmissionError(f"image MIME type is not allowed: {mime_type!r}")
    if len(payload) > image.maximum_file_bytes:
        raise MediaAdmissionError("image file size exceeds the profile limit")
    if mime_type == "image/png" and "png" in image.formats:
        width, height, decoded_bytes = _inspect_png(
            payload,
            maximum_width=image.maximum_width,
            maximum_height=image.maximum_height,
            maximum_pixels=image.maximum_pixels,
            maximum_decoded_bytes=image.maximum_decoded_bytes,
        )
        image_format = "png"
    else:
        raise MediaAdmissionError("image format is not implemented by the admission gate")
    ratio = decoded_bytes / max(len(payload), 1)
    if ratio > image.maximum_decompression_ratio:
        raise MediaAdmissionError("image exceeds the decompression-ratio limit")
    processed_width, processed_height = _scaled_image_dimensions(
        width,
        height,
        factor=processor.image_rescale_factor,
        maximum_upscaled_long_edge=processor.image_rescale_max_upscaled_long_edge,
    )
    processor_tokens = (
        (processed_height + processor.image_patch_size - 1) // processor.image_patch_size
    ) * (processed_width // processor.image_patch_size + 1)
    if processor_tokens > image.maximum_processor_tokens:
        raise MediaAdmissionError("image processor-token count exceeds the profile limit")
    return MediaItem(
        modality="image",
        format=image_format,
        mime_type=mime_type,
        transport_characters=len(encoded),
        file_bytes=len(payload),
        decoded_bytes=decoded_bytes,
        processor_tokens=processor_tokens,
        width=width,
        height=height,
        processed_width=processed_width,
        processed_height=processed_height,
    )


def inspect_audio_part(part: dict[str, Any], settings: MultimodalSettings) -> MediaItem:
    audio = settings.audio
    processor = settings.processor
    if not settings.enabled or not audio.enabled:
        raise MediaAdmissionError("audio input is not enabled by this serving profile")
    if processor is None:
        raise MediaAdmissionError("audio processor configuration is missing")
    input_audio = part.get("input_audio")
    if not isinstance(input_audio, dict):
        raise MediaAdmissionError("input_audio.input_audio must be an object")
    audio_format = input_audio.get("format")
    if not isinstance(audio_format, str) or audio_format.lower() not in audio.formats:
        raise MediaAdmissionError("input_audio format is not allowed")
    encoded = input_audio.get("data")
    if not isinstance(encoded, str):
        raise MediaAdmissionError("input_audio.input_audio.data must be base64 text")
    payload = _decode_base64(
        encoded,
        "input_audio.input_audio.data",
        maximum_characters=audio.maximum_base64_characters,
    )
    if len(payload) > audio.maximum_file_bytes:
        raise MediaAdmissionError("audio file size exceeds the profile limit")
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise MediaAdmissionError("audio format does not match the WAV signature")
    riff_size = struct.unpack("<I", payload[4:8])[0]
    if riff_size + 8 != len(payload):
        raise MediaAdmissionError("WAV RIFF size does not match the file")
    try:
        with wave.open(io.BytesIO(payload), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frames = reader.getnframes()
            compression = reader.getcomptype()
            samples = reader.readframes(frames)
    except (EOFError, wave.Error) as exc:
        raise MediaAdmissionError("WAV file is malformed or unsupported") from exc
    if compression != "NONE":
        raise MediaAdmissionError("WAV must contain uncompressed PCM")
    if sample_rate not in audio.sample_rates_hz:
        raise MediaAdmissionError("WAV sample rate is not allowed")
    if channels not in audio.channels:
        raise MediaAdmissionError("WAV channel count is not allowed")
    if sample_width not in audio.sample_width_bytes:
        raise MediaAdmissionError("WAV sample width is not allowed")
    decoded_bytes = frames * channels * sample_width
    if len(samples) != decoded_bytes:
        raise MediaAdmissionError("WAV sample payload is truncated")
    duration = frames / sample_rate
    if frames > audio.maximum_frames or duration > audio.maximum_duration_seconds:
        raise MediaAdmissionError("WAV duration exceeds the profile limit")
    if decoded_bytes > audio.maximum_decoded_bytes:
        raise MediaAdmissionError("WAV decoded size exceeds the profile limit")
    processor_tokens = (
        frames + processor.audio_samples_per_token - 1
    ) // processor.audio_samples_per_token
    if processor_tokens > audio.maximum_processor_tokens:
        raise MediaAdmissionError("audio processor-token count exceeds the profile limit")
    return MediaItem(
        modality="audio",
        format=audio_format.lower(),
        mime_type="audio/wav",
        transport_characters=len(encoded),
        file_bytes=len(payload),
        decoded_bytes=decoded_bytes,
        processor_tokens=processor_tokens,
        frames=frames,
        duration_seconds=duration,
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
    )


def _content_parts(request: dict[str, Any]) -> list[dict[str, Any]]:
    request_input = request.get("input")
    if not isinstance(request_input, list):
        return []
    parts: list[dict[str, Any]] = []
    for item in request_input:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            parts.extend(part for part in content if isinstance(part, dict))
        elif item.get("type") in {"input_image", "input_audio"}:
            parts.append(item)
    return parts


def _text_utf8_bytes(request: dict[str, Any]) -> int:
    total = 0

    def visit(value: object, *, key: str | None = None, audio_payload: bool = False) -> None:
        nonlocal total
        if isinstance(value, str):
            if key == "image_url" or (audio_payload and key == "data"):
                return
            total += len(value.encode("utf-8"))
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        item_type = value.get("type")
        for child_key, child in value.items():
            visit(
                child,
                key=child_key,
                audio_payload=(
                    audio_payload or (item_type == "input_audio" and child_key == "input_audio")
                ),
            )

    visit(request)
    return total


def validate_responses_media_request(
    request: dict[str, Any], settings: MultimodalSettings
) -> MediaAdmissionReport:
    """Inspect all supported media and reject unreviewed processor controls."""

    if request.get("media_io_kwargs") is not None or request.get("mm_processor_kwargs") is not None:
        raise MediaAdmissionError("per-request media processor overrides are not allowed")
    if request.get("audio") is not None:
        raise MediaAdmissionError("audio generation is outside this service scope")
    output_modalities = request.get("modalities")
    if output_modalities not in (None, ["text"]):
        raise MediaAdmissionError("output modalities must contain only text")
    items: list[MediaItem] = []
    unsupported_media_types = {
        "audio_embeds",
        "audio_url",
        "image_embeds",
        "image_pil",
        "image_url",
        "video_url",
    }
    for part in _content_parts(request):
        part_type = part.get("type")
        if part_type == "input_image":
            items.append(inspect_image_part(part, settings))
        elif part_type == "input_audio":
            items.append(inspect_audio_part(part, settings))
        elif part_type in unsupported_media_types:
            raise MediaAdmissionError(f"media content type is not allowed: {part_type!r}")
    report = MediaAdmissionReport(tuple(items))
    if report.image_count > settings.image.maximum_items:
        raise MediaAdmissionError("image item count exceeds the profile limit")
    if report.audio_count > settings.audio.maximum_items:
        raise MediaAdmissionError("audio item count exceeds the profile limit")
    if report.image_count and report.audio_count and not settings.mixed_media.enabled:
        raise MediaAdmissionError("mixed image and audio input is not enabled")
    if len(report.items) > settings.mixed_media.maximum_total_items:
        raise MediaAdmissionError("total media item count exceeds the profile limit")
    if report.total_decoded_bytes > settings.mixed_media.maximum_total_decoded_bytes:
        raise MediaAdmissionError("total decoded media size exceeds the profile limit")
    if (
        report.image_count
        and report.audio_count
        and report.total_processor_tokens > settings.mixed_media.maximum_processor_tokens
    ):
        raise MediaAdmissionError("mixed-media processor tokens exceed the profile limit")
    if report.items:
        max_output_tokens = request.get("max_output_tokens")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise MediaAdmissionError("media requests require a positive max_output_tokens value")
        conservative_context_tokens = (
            report.total_processor_tokens
            + max_output_tokens
            + settings.minimum_text_context_reserve_tokens
            + _text_utf8_bytes(request)
        )
        if conservative_context_tokens > settings.maximum_context_tokens:
            raise MediaAdmissionError("media request exceeds the multimodal context budget")
    return report
