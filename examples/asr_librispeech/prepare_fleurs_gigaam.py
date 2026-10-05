"""Convert a local FLEURS train parquet to SLAM JSONL; no trainer or audio model changes."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import soundfile as sf
import torch

from slam_llm.models.encoder import GigaAMEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--projector-ds-rate", type=int, default=2)
    args = parser.parse_args()
    source, root = Path(args.parquet), Path(args.output).resolve()
    with source.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != args.sha256:
            raise ValueError("Source parquet SHA256 mismatch")
    if args.projector_ds_rate < 1:
        raise ValueError("Projector downsampling must be positive")
    encoder = GigaAMEncoder.load(OmegaConf.create({"encoder_path": args.encoder_path}))
    root.mkdir(parents=True, exist_ok=False)
    (root / "audio").mkdir()
    counts, seconds = Counter(), Counter()
    source_ids = {"train": set(), "val": set()}
    texts = {"train": set(), "val": set()}
    with (root / "train.jsonl").open("w") as train, (root / "val.jsonl").open("w") as val:
        index = 0
        for batch in pq.ParquetFile(source).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                text = row["raw_transcription"].strip()
                if not text or rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
                    raise ValueError(f"Invalid audio/text at parquet row {index}")
                if len(audio) != row["num_samples"] or not len(audio):
                    raise ValueError(f"Invalid audio length at parquet row {index}")
                # Keep all recordings of the same FLEURS source ID in one split.
                # This salt also preserves the earlier Russian experiment's heldout IDs.
                source_id = str(row["id"])
                heldout = int(hashlib.sha256(("fleurs-ru-" + source_id).encode()).hexdigest(), 16) % 8 == 0
                split = "val" if heldout else "train"
                path = root / "audio" / f"{index:05d}.wav"
                path.write_bytes(row["audio"]["bytes"])
                frames = encoder.get_output_lengths(torch.tensor([len(audio)]))
                slots = int(frames[0]) // args.projector_ds_rate
                if slots < 1:
                    raise ValueError(f"No projected audio frames at parquet row {index}")
                record = {"key": f"fleurs-{source_id}-{index}", "source": str(path),
                          "target": text, "audio_length": slots, "source_len": slots}
                (val if heldout else train).write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                seconds[split] += len(audio) / rate
                source_ids[split].add(source_id)
                texts[split].add(" ".join(row["transcription"].lower().split()))
                index += 1
    if source_ids["train"] & source_ids["val"] or texts["train"] & texts["val"]:
        raise ValueError("Training/validation source or transcript overlap")
    report = {"source_sha256": args.sha256, "counts": dict(counts),
              "hours": {k: v / 3600 for k, v in seconds.items()},
              "projector_ds_rate": args.projector_ds_rate,
              "validation": "Deterministic source-ID holdout from train; not official FLEURS test"}
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
