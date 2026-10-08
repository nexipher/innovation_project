"""Tests for the LoRA trainer's CPU paths (G4-g)."""

import json

import pytest
import torch
from PIL import Image

from config import PROJECT_ROOT

from scripts.train_lora_g4 import (
    TrajectoryDataset,
    build_messages,
    load_training_samples,
    sampling_weights,
    split_answer,
    summarise,
    system_prompt_for,
)


def _sample(sample_id="s1", tools=("noise",), truth="Fake",
            with_artifacts=True, answer="<observation>\n...</observation>",
            image="dataset/Real/x.jpg", artifacts=("traces/evidence/s/region.png",
                                                  "traces/evidence/s/map.png")):
    metadata = {"tools_served": list(tools), "treatment": "png", "policy": "noise"}
    conversations = [{"from": "user", "value": "<image>\n请分析这张图像的真实性。"}]
    if tools:
        conversations.append({"from": "gpt", "value": "<planning>...</planning>\n"
                                                   "<call_noise>[1, 2, 3, 4]</call_noise>"})
        evidence_turn = {"from": "user", "value": '{"evidence_id": "E-1"}'}
        if with_artifacts:
            evidence_turn["image_paths"] = list(artifacts)
        conversations.append(evidence_turn)
    conversations.append({"from": "gpt", "value": answer})
    return {"id": sample_id, "image_path": image,
            "ground_truth": truth, "type": "positive",
            "conversations": conversations, "evidence_chain": [], "metadata": metadata}


class TestLoading:
    def test_the_provisional_set_reads_the_final_v2_buckets(self, tmp_path):
        for name in ("sft_tool_positive", "sft_no_tool_positive", "sft_honest_abstention"):
            (tmp_path / f"{name}.json").write_text(json.dumps([_sample(name)]),
                                                   encoding="utf-8")
        samples = load_training_samples(from_final_v2=True, final_v2_dir=str(tmp_path))
        assert len(samples) == 3

    def test_the_reviewed_set_is_the_default_source(self, tmp_path):
        (tmp_path / "sft_accepted.json").write_text(json.dumps([_sample()]), encoding="utf-8")
        (tmp_path / "sft_rejected_reviewed.json").write_text(json.dumps([_sample("bad")]),
                                                             encoding="utf-8")
        samples = load_training_samples(reviewed_dir=str(tmp_path))
        assert [s["id"] for s in samples] == ["s1"]

    def test_a_missing_directory_yields_nothing(self, tmp_path):
        assert load_training_samples(reviewed_dir=str(tmp_path / "nope")) == []


class TestSystemPrompt:
    def test_a_tool_free_sample_gets_the_baseline_prompt(self):
        assert "NO access to external forensic tools" in \
            system_prompt_for(_sample(tools=()))

    def test_a_tool_sample_lists_only_its_own_tools(self):
        prompt = system_prompt_for(_sample(tools=("noise", "jpeg")))
        actions = prompt.split("variations):")[1].split("FORBIDDEN")[0]
        assert "- <call_noise>" in actions and "- <call_jpeg>" in actions
        assert "- <call_freq>" not in actions


class TestSplitAnswer:
    def test_it_separates_prompt_from_answer(self):
        prompt, answer = split_answer(_sample())
        assert answer["value"].startswith("<observation>")
        assert all(turn["from"] != "gpt" or "<call_" in turn["value"] for turn in prompt)

    def test_a_conversation_not_ending_in_an_answer_is_refused(self):
        sample = _sample()
        sample["conversations"] = sample["conversations"][:-1]
        with pytest.raises(ValueError, match="does not end with the answer"):
            split_answer(sample)


class TestSamplingWeights:
    def test_weights_offset_the_class_imbalance(self):
        samples = [_sample("a", truth="Fake"), _sample("b", truth="Fake"),
                   _sample("c", truth="Real")]
        weights, labels = sampling_weights(samples)
        assert labels == {"Fake": 2, "Real": 1}
        assert weights == [1.0, 1.0, 2.0]

    def test_balancing_can_be_turned_off(self):
        samples = [_sample("a"), _sample("b", truth="Real")]
        weights, _ = sampling_weights(samples, balance=False)
        assert weights == [1.0, 1.0]

    def test_an_empty_set_is_handled(self):
        assert sampling_weights([]) == ([], {})


class TestSummarise:
    def test_it_counts_policies_and_treatments(self):
        summary = summarise([_sample("a"), _sample("b", tools=())],
                            [1.0, 1.0])
        assert summary["samples"] == 2
        assert summary["policies"] == {"noise": 1, "no-tool": 1}
        assert summary["labels"] == {"Fake": 2}


class TestMessageAssembly:
    def test_the_prompt_is_the_answer_minus_the_last_turn(self):
        prompt_messages, full_messages, _ = build_messages(_sample())
        assert len(full_messages) == len(prompt_messages) + 1
        assert full_messages[-1]["role"] == "assistant"

    def test_the_image_marker_is_resolved_to_the_sample_image(self):
        """Without this the model would train on a conversation with no image."""
        _, full_messages, _ = build_messages(_sample(), load_images=False)
        first_user = next(m for m in full_messages if m["role"] == "user")
        blocks = [b for b in first_user["content"] if b.get("type") == "image"]
        assert blocks and blocks[0]["path"] == "dataset/Real/x.jpg"
        assert "<image>" not in first_user["content"][-1]["text"]

    def test_evidence_artifacts_are_attached_in_order(self):
        _, full_messages, _ = build_messages(_sample(), load_images=False)
        paths = [b["path"] for m in full_messages if isinstance(m["content"], list)
                 for b in m["content"] if b.get("type") == "image"]
        assert paths == ["dataset/Real/x.jpg",
                         "traces/evidence/s/region.png",
                         "traces/evidence/s/map.png"]

    def test_a_tool_free_sample_has_only_the_sample_image(self):
        _, full_messages, _ = build_messages(_sample(tools=()), load_images=False)
        paths = [b["path"] for m in full_messages if isinstance(m["content"], list)
                 for b in m["content"] if b.get("type") == "image"]
        assert paths == ["dataset/Real/x.jpg"]


class _StubProcessor:
    """
    A processor shaped like Qwen's, so the dataset's tensor handling is
    testable without loading the real one.

    `pixel_values` is (patches, dim) and `image_grid_thw` is (images, 3) — both
    *without* a batch dimension, which is what makes them easy to break: an
    unconditional `value[0]` silently trains on the first image only.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """Ten tokens of prompt, plus five if the answer is included."""
        text = "x" * 10
        if messages and messages[-1]["role"] == "assistant":
            text += "y" * 5
        return text

    def __call__(self, text, images=None, return_tensors=None, padding=None):
        length = len(text[0])
        ids = torch.arange(length).unsqueeze(0)
        out = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if images:
            out["pixel_values"] = torch.zeros((len(images) * 4, 3))
            out["image_grid_thw"] = torch.tensor([[1, 2, 2]] * len(images))
        return out


class TestDataset:
    """The dataset opens real files, so these use real (tiny) images."""

    def _dataset(self, tmp_path):
        image = tmp_path / "source.png"
        Image.new("RGB", (16, 16), (10, 20, 30)).save(image)
        artifacts = []
        for name in ("region", "map"):
            path = tmp_path / f"{name}.png"
            Image.new("RGB", (8, 8), (30, 20, 10)).save(path)
            artifacts.append(str(path))
        sample = _sample(image=str(image), artifacts=tuple(artifacts))
        return TrajectoryDataset([sample], _StubProcessor())

    def test_it_loads_the_sample_and_its_artifacts(self, tmp_path):
        item = self._dataset(tmp_path)[0]
        assert item["image_grid_thw"].shape[0] == 3

    def test_the_prompt_is_masked_and_the_answer_is_supervised(self, tmp_path):
        item = self._dataset(tmp_path)[0]
        labels = item["labels"]
        assert (labels == -100).sum() > 0
        assert (labels != -100).sum() > 0
        assert item["answer_tokens"] > 0

    def test_multi_image_tensors_are_not_squeezed(self, tmp_path):
        """The regression: [0] on these tensors drops every image but the first."""
        item = self._dataset(tmp_path)[0]
        assert item["image_grid_thw"].shape[0] == 3      # original + 2 artifacts
        assert item["pixel_values"].shape[0] == 12       # 3 images x 4 patches

    def test_the_text_tensors_lose_their_batch_dimension(self, tmp_path):
        item = self._dataset(tmp_path)[0]
        assert item["input_ids"].dim() == 1
        assert item["attention_mask"].dim() == 1
        assert item["labels"].shape == item["input_ids"].shape

    def test_length_is_reported(self, tmp_path):
        item = self._dataset(tmp_path)[0]
        assert item["prompt_tokens"] + item["answer_tokens"] == item["input_ids"].shape[0]
