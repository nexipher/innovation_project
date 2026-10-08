"""Tests for the LoRA trainer's CPU paths (G4-g)."""

import glob
import json
import os
import re

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from config import PROJECT_ROOT

from scripts.train_lora_g4 import (
    TrajectoryDataset,
    assistant_spans,
    build_messages,
    compute_masked_loss,
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

    def test_a_tool_sample_gets_the_shipped_prompt(self):
        """Not the generator's per-policy subset: inference never sends those."""
        from mllm.message_builder import FORENSIC_SYSTEM_PROMPT

        prompt = system_prompt_for(_sample(tools=("noise",)))
        assert prompt == FORENSIC_SYSTEM_PROMPT
        actions = prompt.split("variations):")[1].split("FORBIDDEN")[0]
        assert "- <call_noise>" in actions and "- <call_freq>" in actions


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


class _StubTokenizer:
    """The marker lookups the label builder makes, and nothing else."""

    IDS = {"<|im_start|>": 151644, "<|im_end|>": 151645,
           "assistant": 77091, "user": 872}

    def convert_tokens_to_ids(self, token):
        return self.IDS[token]


class _StubProcessor:
    """
    A processor shaped like Qwen's, so the dataset's tensor handling is
    testable without loading the real one.

    The template emits real turn markers, because the labels are built by
    finding them: user turns are ten `x`, assistant turns five `y`, and the
    markers map to the ids the scanner looks for.

    `pixel_values` is (patches, dim) and `image_grid_thw` is (images, 3) — both
    *without* a batch dimension, which is what makes them easy to break: an
    unconditional `value[0]` silently trains on the first image only.
    """

    def __init__(self):
        self.tokenizer = _StubTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            body = "y" * 5 if message["role"] == "assistant" else "x" * 10
            parts.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
        return "".join(parts)

    def __call__(self, text, images=None, return_tensors=None, padding=None):
        pieces = re.findall(r"<\|im_start\|>|<\|im_end\|>|assistant|user|x|y", text[0])
        ids = torch.tensor([[self.tokenizer.IDS.get(piece, 500 + ord(piece[0]) % 100)
                             for piece in pieces]])
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
        assert item["supervised_tokens"] > 0

    def test_the_tool_call_turn_is_supervised_too(self, tmp_path):
        """
        The regression this exists for: masking everything before the final
        answer leaves the call turn unsupervised, and a model that was never
        trained to emit `<call_*>` answers straight from the question — which
        is what the first provisional adapter did, in 126 of 127 sessions.
        """
        item = self._dataset(tmp_path)[0]          # tool sample: call + answer
        assert item["supervised_turns"] == 2
        supervised = item["labels"][item["labels"] != -100]
        # Both assistant turns in the stub are five `y` tokens plus `<|im_end|>`.
        assert supervised.numel() == 2 * (5 + 1)

    def test_a_tool_free_sample_supervises_only_the_answer(self, tmp_path):
        image = tmp_path / "source.png"
        Image.new("RGB", (16, 16), (10, 20, 30)).save(image)
        dataset = TrajectoryDataset([_sample(tools=(), image=str(image))],
                                    _StubProcessor())
        assert dataset[0]["supervised_turns"] == 1

    def test_multi_image_tensors_are_not_squeezed(self, tmp_path):
        """The regression: [0] on these tensors drops every image but the first."""
        item = self._dataset(tmp_path)[0]
        assert item["image_grid_thw"].shape[0] == 3      # original + 2 artifacts
        assert item["pixel_values"].shape[0] == 12       # 3 images x 4 patches

    def test_the_text_tensors_keep_their_batch_dimension(self, tmp_path):
        """The vision position computation indexes input_ids.shape[1]."""
        item = self._dataset(tmp_path)[0]
        assert item["input_ids"].dim() == 2
        assert item["attention_mask"].dim() == 2
        assert item["labels"].shape == item["input_ids"].shape

class TestAssistantSpans:
    """The label builder, exercised on ids rather than through a processor."""

    def test_it_spans_every_assistant_turn(self):
        ids = [151644, 872, 5, 5, 151645,          # <|im_start|>user ... <|im_end|>
               151644, 77091, 7, 7, 7, 151645,     # assistant turn
               151644, 872, 5, 151645,             # user turn
               151644, 77091, 9, 151645]           # assistant turn
        assert assistant_spans(ids, _StubTokenizer()) == [(7, 11), (17, 19)]

    def test_a_conversation_without_markers_yields_nothing(self):
        assert assistant_spans([1, 2, 3], _StubTokenizer()) == []

    def test_an_unterminated_turn_still_ends_at_the_sequence(self):
        ids = [151644, 77091, 7, 8]
        assert assistant_spans(ids, _StubTokenizer()) == [(2, 4)]


class _StubVLModel:
    """
    The two things the loss path touches: a callable returning hidden states,
    and the language-model head.

    `hidden_states` is a tuple whose *last* entry is the only one carrying the
    token embeddings — the loss has to read the last layer, and a version that
    read `[0]` would score noise.
    """

    def __init__(self, vocab=7, hidden=6):
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs["input_ids"]
        buried = torch.zeros_like(self.embed(ids))

        class _Output:
            pass

        output = _Output()
        output.hidden_states = (buried, self.embed(ids))
        return output


def _batch(ids, labels, **extra):
    batch = {
        "input_ids": torch.tensor([ids]),
        "labels": torch.tensor([labels]),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
    }
    batch.update(extra)
    return batch


class TestMaskedLoss:
    """
    The model's own loss projects every position through a 152k-token head;
    this one projects only the supervised positions (the memory fix that made
    a 2.4k-token sample fit on 24 GB).  These pin the objective so the saving
    cannot quietly change what is being optimised.
    """

    def test_it_equals_cross_entropy_on_the_answer_positions(self):
        model = _StubVLModel()
        ids = [1, 2, 3, 4, 5]
        labels = [-100, -100, 3, 4, 5]
        loss = compute_masked_loss(model, _batch(ids, labels))

        hidden = model.embed(torch.tensor([ids]))[:, :-1, :]
        shifted = torch.tensor([labels])[:, 1:]
        mask = shifted != -100
        expected = F.cross_entropy(model.lm_head(hidden[mask]).float(), shifted[mask])
        assert torch.allclose(loss, expected)

    def test_a_masked_position_does_not_reach_the_loss(self):
        """Only positions whose *next* token is supervised may contribute."""
        model = _StubVLModel()
        labels = [-100, 3, -100, 4, 5]
        baseline = compute_masked_loss(model, _batch([1, 2, 3, 4, 5], labels))
        perturbed = compute_masked_loss(model, _batch([1, 6, 3, 4, 5], labels))
        assert torch.allclose(baseline, perturbed)

    def test_the_head_only_sees_the_supervised_positions(self):
        model = _StubVLModel(vocab=7, hidden=6)
        seen = {}

        def record(module, args):
            seen["shape"] = tuple(args[0].shape)
            return None   # a hook returning a tuple would *replace* the inputs

        handle = model.lm_head.register_forward_pre_hook(record)
        try:
            compute_masked_loss(model, _batch([1, 2, 3, 4, 5], [-100, -100, 3, 4, 5]))
        finally:
            handle.remove()
        assert seen["shape"] == (3, 6)   # three supervised positions, not five

    def test_the_multimodal_inputs_reach_the_model(self):
        model = _StubVLModel()
        pixels = torch.zeros(4, 3)
        grid = torch.tensor([[1, 2, 2]])
        compute_masked_loss(model, _batch([1, 2, 3], [-100, 2, 3],
                                          pixel_values=pixels, image_grid_thw=grid))
        call = model.calls[-1]
        assert call["pixel_values"] is pixels and call["image_grid_thw"] is grid
        assert call["output_hidden_states"] is True
        assert call["use_cache"] is False

    def test_the_loss_stays_attached_to_the_graph(self):
        model = _StubVLModel()
        loss = compute_masked_loss(model, _batch([1, 2, 3], [-100, 2, 3]))
        loss.backward()
        assert model.lm_head.weight.grad is not None
        assert model.embed.weight.grad is not None


def _load_json(*parts):
    with open(os.path.join(PROJECT_ROOT, *parts), encoding="utf-8") as handle:
        return json.load(handle)


class TestTrainingLeak:
    """
    The trial trains on sources from the `train` split and is measured on the
    calibration sources, which the split excludes — a rebuild that pulled
    training data from the wrong pool would produce a gain that is really
    memorisation.  These read the real artifacts, so they fail if the two
    pools ever touch.
    """

    def _training_source_paths(self, split):
        """Training sample -> the dataset file it was rendered from."""
        by_stem = {key.replace("/", "_"): value for key, value in split["sources"].items()}
        paths, unresolved = set(), []
        for name in glob.glob(os.path.join(PROJECT_ROOT, "sft_data/train/final_v2/sft_*.json")):
            for sample in json.load(open(name, encoding="utf-8")):
                stem = re.sub(r"_(native|png|jpe?g_q\d+)\.[A-Za-z]+$", "",
                              os.path.basename(sample["image_path"]))
                if stem not in by_stem:
                    unresolved.append(stem)
                    continue
                paths.add((by_stem[stem]["path"], by_stem[stem]["split"]))
        return paths, unresolved

    def test_every_training_source_resolves_into_the_split(self):
        split = _load_json("sft_data", "split_v2.json")
        paths, unresolved = self._training_source_paths(split)
        assert unresolved == []
        assert paths

    def test_training_sources_are_all_in_the_train_split(self):
        split = _load_json("sft_data", "split_v2.json")
        paths, _ = self._training_source_paths(split)
        assert {stage for _, stage in paths} == {"train"}

    def test_no_training_source_is_a_calibration_source(self):
        """The eval pool is the 98 held-out sources the split table never sees."""
        split = _load_json("sft_data", "split_v2.json")
        training = {path for path, _ in self._training_source_paths(split)[0]}
        manifest = _load_json("calibration", "set", "manifest.json")
        calibration = {sample["source_path"] for sample in manifest["samples"]}
        assert training and calibration
        assert training & calibration == set()
