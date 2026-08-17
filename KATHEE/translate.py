# translate.py (reusable for baseline AND fine-tuned inference)
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from config import *

def load_model(model_name, adapter_path=None):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to("cuda").eval()
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).to("cuda").eval()
    return tok, model

ip = IndicProcessor(inference=True)

@torch.no_grad()
def translate(sentences, tok, model,
              num_beams=NUM_BEAMS, length_penalty=LENGTH_PENALTY, batch_size=BATCH_SIZE):
    out = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        pre = ip.preprocess_batch(batch, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
        enc = tok(pre, truncation=True, padding=True, max_length=256,
                  return_tensors="pt").to("cuda")
        gen = model.generate(**enc, num_beams=num_beams,
                             length_penalty=length_penalty,
                             max_new_tokens=MAX_NEW_TOKENS,
                             num_return_sequences=1,
                             no_repeat_ngram_size=4)
        with tok.as_target_tokenizer():
            dec = tok.batch_decode(gen.detach().cpu().tolist(),
                                   skip_special_tokens=True)
        out.extend(ip.postprocess_batch(dec, lang=TGT_LANG))
    return out
