import sys
from types import SimpleNamespace

import pytest

from turboforge.domain import ForgeConfig
from turboforge.evaluation import AUTO_EVALUATION_TOKENS, WIKITEXT_REVISION, evaluation_text


def test_automatic_evaluation_is_fixed_test_split(tmp_path, monkeypatch):
    requests = []
    def load(*args, **kwargs):
        requests.append((args, kwargs))
        return [{"text": "独立测试材料"}, {"text": ""}, {"text": "第二段"}]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load))
    text, identity = evaluation_text({"auto_quality": True}, tmp_path)
    assert text == "独立测试材料\n\n第二段"
    assert requests[0][1] == {"split": "test", "revision": WIKITEXT_REVISION}
    assert identity["token_limit"] == AUTO_EVALUATION_TOKENS
    assert (tmp_path / "evaluation.json").is_file()


@pytest.mark.parametrize("changes", [
    {"target": "auto"}, {"target": "hf_packed_import"}, {"run_benchmark": False},
    {"benchmark": {"auto_quality": True, "perplexity_text": "a.txt"}},
    {"benchmark": {"auto_quality": "yes"}},
])
def test_invalid_automatic_quality_rejected_before_download(changes):
    values = {"input_path": "model", "output_dir": "new-output", "target": "hf_gptq_4",
              "run_benchmark": True, "benchmark": {"auto_quality": True, "max_ppl_relative_increase": .1}, **changes}
    with pytest.raises(ValueError):
        ForgeConfig.from_dict(values)


def test_automatic_quality_needs_no_manual_files():
    config = ForgeConfig.from_dict({"input_path": "model", "output_dir": "new-output", "target": "hf_gptq_4",
        "run_benchmark": True, "benchmark": {"auto_quality": True, "max_ppl_relative_increase": .1}})
    assert config.benchmark["auto_quality"] is True
