"""Contract tests using real, tiny GigaAM and Qwen modules; no weight downloads."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM

GigaAM = pytest.importorskip("gigaam.model").GigaAM
from slam_llm.models.encoder import GigaAMEncoder
from slam_llm.models.projector import EncoderProjectorConcat
from slam_llm.models.slam_model import slam_model, setup_encoder


@pytest.fixture
def acoustic_model():
    torch.manual_seed(17)
    return GigaAM(OmegaConf.create({
        "preprocessor": {
            "_target_": "gigaam.preprocess.FeatureExtractor",
            "sample_rate": 16000, "features": 16,
        },
        "encoder": {
            "_target_": "gigaam.encoder.ConformerEncoder",
            "feat_in": 16, "n_layers": 1, "d_model": 32,
            "n_heads": 4, "ff_expansion_factor": 2,
            "subsampling_factor": 4, "conv_kernel_size": 3,
            "pos_emb_max_len": 128, "flash_attn": False,
        },
    })).eval()


def test_native_features_and_frame_lengths(acoustic_model):
    audio = torch.randn(2, 3200)
    lengths = torch.tensor([3200, 1600])
    padding = torch.arange(3200)[None] >= lengths[:, None]
    audio[padding] = 0
    encoder = GigaAMEncoder(acoustic_model).eval()
    with torch.no_grad():
        native, native_lengths = acoustic_model(audio, lengths)
        features, valid = encoder.extract_features(audio, padding)
    torch.testing.assert_close(features[valid], native.transpose(1, 2)[valid])
    torch.testing.assert_close(valid.sum(1), native_lengths.long())
    torch.testing.assert_close(encoder.get_output_lengths(lengths), native_lengths)
    assert torch.count_nonzero(features[~valid]) == 0


def test_padding_is_optional(acoustic_model):
    encoder = GigaAMEncoder(acoustic_model).eval()
    audio = torch.randn(1, 1600)
    features, valid = encoder.extract_features(audio)
    assert features.shape[:2] == valid.shape
    assert valid.all()


def test_loader_uses_native_checkpoint_api_and_drops_asr_head(acoustic_model, monkeypatch):
    calls = []
    acoustic_model.head = torch.nn.Linear(32, 4)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return acoustic_model

    monkeypatch.setattr("gigaam.load_model", load)
    encoder = setup_encoder(
        SimpleNamespace(freeze_encoder=True, enable_fsdp=False, enable_ddp=False),
        SimpleNamespace(encoder_name="gigaam", encoder_path="/models/local.ckpt"),
    )
    assert calls == [("/models/local.ckpt", {
        "device": torch.device("cuda:2"),
        "fp16_encoder": True, "use_flash": True,
    })]
    assert not any(p.requires_grad for p in encoder.parameters())
    assert not hasattr(encoder.model, "head")


@pytest.mark.parametrize("wrong_slots", [False, True])
def test_slam_qwen_projector_backward_and_length_guard(acoustic_model, wrong_slots):
    config = OmegaConf.create({
        "encoder_name": "gigaam", "encoder_projector": "linear",
        "encoder_projector_ds_rate": 2, "encoder_dim": 32, "llm_dim": 32,
    })
    encoder = GigaAMEncoder(acoustic_model).requires_grad_(False)
    llm = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )).requires_grad_(False)
    projector = EncoderProjectorConcat(config)
    model = slam_model(encoder, llm, projector, None,
                       OmegaConf.create({"freeze_encoder": True}), config)
    audio = torch.randn(2, 3200)
    lengths = torch.tensor([3200, 1600])
    audio_mask = torch.arange(3200)[None] < lengths[:, None]
    slots = encoder.get_output_lengths(lengths).long() // 2
    if wrong_slots:
        slots = slots + 1
    size = int(slots.max()) + 3
    ids = torch.ones(2, size, dtype=torch.long)
    modality = torch.arange(size)[None] < slots[:, None]
    ids[modality] = -1
    labels = torch.full_like(ids, -100)
    labels[:, -2:] = 3
    batch = dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids),
                 audio=audio, audio_mask=audio_mask, modality_mask=modality)
    if wrong_slots:
        with pytest.raises(ValueError, match="placeholders must match"):
            model(**batch)
        return
    outputs, _ = model(**batch)
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    assert all(p.grad is None for p in encoder.parameters())
    assert all(p.grad is None for p in llm.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in projector.parameters())
    assert projector.linear1.weight.grad.abs().sum() > 0
