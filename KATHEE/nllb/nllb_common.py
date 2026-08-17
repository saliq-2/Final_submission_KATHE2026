# nllb/nllb_common.py
# Shared --model/--epochs CLI parsing and model-tag naming, used by
# finetune_nllb.py, eval_nllb.py, and make_dev_submission_nllb.py so the
# three stay in sync on adapter naming (all three need to agree on the
# checkpoint path for a given model+branch, or eval/submission will silently
# load the wrong adapter).
import sys

MODEL_ALIASES = {
    "600m": "facebook/nllb-200-distilled-600M",
    "1.3b": "facebook/nllb-200-1.3B",
    "1.3b-distilled": "facebook/nllb-200-distilled-1.3B",
    "3.3b": "facebook/nllb-200-3.3B",
}


def get_arg(flag, default=None, cast=str):
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return cast(sys.argv[idx + 1])
    return default


def resolve_model_name(argv=None):
    """--model accepts a short alias (see MODEL_ALIASES) or any full HF
    hub id. Defaults to 600M (the one already trained/evaluated)."""
    arg = get_arg("--model", "600m")
    return MODEL_ALIASES.get(arg.lower(), arg)


def model_tag(model_name):
    """Short tag for checkpoint/adapter naming, e.g. 'distilled-600M', '1.3B'."""
    return model_name.split("/")[-1].replace("nllb-200-", "")
