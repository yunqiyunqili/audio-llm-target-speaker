"""Check the cloned env envs/qwen_omni_lora (docs/lora_slu_plan_v2.md §10). CPU only, no weights loaded.
1. import test: torch, transformers, accelerate, peft versions.
2. minimal LoRA forward/backward on a tiny randomly initialised Qwen2 causal LM (q,k,v,o, r=16, alpha=32, dropout 0.05):
   at init the LoRA model output equals the base output (B = 0); only LoRA params are trainable; one backward step gives grads.
3. target-module names on the real Qwen2.5-Omni thinker (meta device, config only, no weights): count q/k/v/o_proj Linear
   modules inside the thinker language model and inside the audio encoder (the latter must not be targeted).
Usage: python -I -B scripts/lora/peft_env_check.py QWEN_SNAPSHOT_DIR
"""
import json
import sys

import torch


def main(qdir):
    import accelerate
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model
    out = dict(torch=torch.__version__, transformers=transformers.__version__, accelerate=accelerate.__version__, peft=peft.__version__)
    print('IMPORT_OK', json.dumps(out), flush=True)
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(2026)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    base = Qwen2ForCausalLM(cfg).eval()
    x = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        y0 = base(input_ids=x).logits.clone()
    lc = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM')
    m = get_peft_model(base, lc)
    m.eval()
    with torch.no_grad():
        y1 = m(input_ids=x).logits
    same = torch.allclose(y0, y1, atol=1e-6)
    tr = [n for n, p in m.named_parameters() if p.requires_grad]
    m.train(); loss = m(input_ids=x, labels=x).loss; loss.backward()
    g = all(p.grad is not None for n, p in m.named_parameters() if p.requires_grad)
    ok_names = all('lora_' in n for n in tr)
    out.update(tiny_init_equal=bool(same), trainable_tensors=len(tr), trainable_params=sum(p.numel() for p in m.parameters() if p.requires_grad),
               only_lora_trainable=ok_names, grads_ok=bool(g), loss=float(loss))
    print('TINY', json.dumps(out), flush=True)
    assert same and ok_names and g and len(tr) == 2 * 4 * 2
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen2_5OmniThinkerForConditionalGeneration
    full = AutoConfig.from_pretrained(qdir, local_files_only=True)
    with init_empty_weights():
        th = Qwen2_5OmniThinkerForConditionalGeneration(full.thinker_config)
    names = [n for n, mod in th.named_modules() if isinstance(mod, torch.nn.Linear) and n.split('.')[-1] in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
    lm = [n for n in names if n.startswith('model.')]
    au = [n for n in names if n.startswith('audio_tower.')]
    vi = [n for n in names if n.startswith('visual.')]
    rest = [n for n in names if n not in lm + au + vi]
    out.update(thinker_lm_qkvo=len(lm), audio_tower_qkvo=len(au), visual_qkvo=len(vi), other_qkvo=rest[:5], lm_example=lm[:4], audio_example=au[:2],
               lm_layers=full.thinker_config.text_config.num_hidden_layers)
    print('THINKER', json.dumps(out), flush=True)
    print('PEFT_ENV_CHECK_PASS', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
