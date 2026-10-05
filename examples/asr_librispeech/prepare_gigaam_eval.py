"""Adapt a GigaAM evaluation TSV to SLAM JSONL without changing audio or labels."""
import argparse
import csv
import json
from pathlib import Path

from omegaconf import OmegaConf
import soundfile as sf
import torch

from slam_llm.models.encoder import GigaAMEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projector-ds-rate", type=int, default=2)
    args = parser.parse_args()
    if args.projector_ds_rate < 1:
        raise ValueError("Projector downsampling must be positive")
    encoder = GigaAMEncoder.load(OmegaConf.create({"encoder_path": args.encoder_path}))
    records, paths = [], set()
    with args.manifest.open() as source:
        for row in csv.DictReader(source, delimiter="\t"):
            path = (args.manifest.parent / row["path"]).resolve(strict=True)
            info = sf.info(path)
            target = row["transcription"]
            if path in paths or not target.strip():
                raise ValueError(f"Duplicate audio or empty transcript: {path}")
            if info.samplerate != 16000 or info.channels != 1 or info.frames < 1:
                raise ValueError(f"Expected nonempty mono 16 kHz audio: {path}")
            slots = int(encoder.get_output_lengths(torch.tensor([info.frames]))[0]) // args.projector_ds_rate
            if slots < 1:
                raise ValueError(f"No projected audio frames: {path}")
            records.append({"key": str(len(records)), "source": str(path),
                            "target": target, "audio_length": slots, "source_len": slots})
            paths.add(path)
    if not records:
        raise ValueError("Empty evaluation manifest")
    with args.output.open("x") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Exported {len(records)} evaluation records to {args.output}")


if __name__ == "__main__":
    main()
