#!/usr/bin/env python3
"""LoRA baseline at the CRN's parameter budget.

Mirrors train_prajna2b.py exactly: same data (error_correction_pairs_v2.json),
same step semantics (micro-steps, batch=1, GRAD_ACCUM=8), same objectives
(answer-only weighted CE with anchor 4x / eos 5x / padding 0.25 weighting;
reference-free DPO ranking, beta=0.1), same LRs and schedule quirks.
Only the adapter differs: standard LoRA (rank 19, 8 depths, all-linear,
6,624,768 trainable params) vs the 6,721,444-param CRN.

Usage:
  HF_HOME="/Volumes/KIOXIA 1TB/huggingface_cache" TMPDIR="/Volumes/KIOXIA 1TB/tmp" \
    python3 prajna-phase2/src/train_lora_baseline.py
"""
import os, sys, json, time, gc, traceback
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SRC))
DATA = os.environ.get('EC_DATA', f'{ROOT}/prajna/data/error_correction_pairs_v2.json')
CKPT = os.environ.get('LORA_CKPT_DIR', f'{ROOT}/prajna/checkpoints')
STATE = os.environ.get('LORA_STATE', f'{CKPT}/state_lora_baseline.json')
DEVICE = os.environ.get('LORA_DEVICE', 'mps')
DPO_DEVICE = os.environ.get('LORA_DPO_DEVICE', DEVICE)  # CPU avoids MPS double-forward deadlock
MAXLEN = int(os.environ.get('LORA_MAXLEN', '96'))
SFT_STEPS = int(os.environ.get('LORA_SFT_STEPS', '3000'))
DPO_STEPS = int(os.environ.get('LORA_DPO_STEPS', '3000'))
GRAD_ACCUM = int(os.environ.get('LORA_GRAD_ACCUM', '8'))
SFT_LR = 3e-4
DPO_LR = 5e-6
DPO_BETA = 0.1
ANCHOR_WEIGHT = float(os.environ.get('ANCHOR_WEIGHT', '4.0'))
EOS_WEIGHT = 5.0
MAX_GRAD_NORM = 1.0
SAVE_EVERY = int(os.environ.get('LORA_SAVE_EVERY', '500'))
LOG_EVERY = 20
RANK = int(os.environ.get('LORA_RANK', '19'))
DEPTHS = [3, 7, 11, 15, 19, 23, 27, 31]


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {}


def save_state(s):
    json.dump(s, open(STATE, 'w'))


class ECSFTDataset(Dataset):
    """Identical encoding to train_prajna2b.py:ECSFTDataset (anchor 4x at the
    answer-start, eos boost, padding 0.25, neighbor 0.5)."""

    def __init__(self, path, tok, ml):
        with open(path) as f:
            self.p = json.load(f)
        self.tok, self.ml = tok, ml

    def __len__(self):
        return len(self.p)

    def _encode(self, s):
        text = f"{s['prompt']}: {s['chosen']}{self.tok.eos_token}"
        enc = self.tok(text, truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        ids = enc['input_ids'].squeeze()
        labels = ids.clone()
        labels[enc['attention_mask'].squeeze() == 0] = -100
        pre_ids = self.tok(f"{s['prompt']}: ", truncation=True, max_length=self.ml,
                           return_tensors='pt')['input_ids'].squeeze()
        labels[:pre_ids.shape[0]] = -100
        pre_len = pre_ids.shape[0]
        weights = torch.ones(self.ml, dtype=torch.float32) * 0.25
        weights[enc['attention_mask'].squeeze() == 1] = 1.0
        if pre_len < ids.shape[0]:
            labels[pre_len - 1] = ids[pre_len]
            weights[pre_len - 1] = ANCHOR_WEIGHT
            if pre_len >= 3:
                weights[pre_len - 2] = 0.5
                weights[pre_len - 3] = 0.5
        else:
            weights[pre_len - 1] = ANCHOR_WEIGHT
        return ids, labels, weights

    def __getitem__(self, i):
        s = self.p[i]
        ids, labels, weights = self._encode(s)
        return {'input_ids': ids, 'labels': labels, 'weights': weights}


class ECDataset(Dataset):
    """Identical to train_prajna2b.py:ECDataset (chosen/rejected encoded bare,
    no prompt prefix — mirrors the CRN trainer's DPO quirk)."""

    def __init__(self, path, tok, ml):
        with open(path) as f:
            self.p = json.load(f)
        self.tok, self.ml = tok, ml

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        s = self.p[i]
        c = self.tok(s['chosen'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        r = self.tok(s['rejected'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        return {'chosen_ids': c['input_ids'].squeeze(), 'rejected_ids': r['input_ids'].squeeze()}


def weighted_ce(logits, labels, weights, eos_id):
    # fused fp16 CE per position (same kernel the CRN trainer uses; the
    # fp32 log_softmax over the 262K vocab is ~10x slower on MPS)
    tgt = labels[:, 1:]
    per_token = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                tgt.reshape(-1), ignore_index=-100, reduction='none')
    per_token = per_token.reshape(labels.shape[0], -1)
    w = weights[:, 1:] if weights.shape[1] == labels.shape[1] else weights
    eos_mask = (labels[:, 1:] == eos_id).float()
    w = w * (1.0 + (EOS_WEIGHT - 1.0) * eos_mask)
    eff = w * (labels[:, 1:] != -100).float()
    return (per_token * eff).sum() / eff.sum().clamp(min=1e-6)


@torch.no_grad()
def noop():
    pass


def seq_logps(logits, ids):
    # sum logp = -sum CE over all positions (pads included, mirroring the
    # CRN trainer's _get_batch_logps which has no pad mask)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                         ids[:, 1:].reshape(-1), reduction='none')
    return -ce.sum()


def main():
    t0 = time.time()
    state = load_state()
    dpo_resumed = state.get('dpo_step', 0) > 0
    tok = AutoTokenizer.from_pretrained('google/gemma-4-E2B')
    depths_re = '|'.join(str(d) for d in DEPTHS)
    targets = (rf'model\.language_model\.layers\.({depths_re})\.'
               rf'(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$')
    cfg = LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05,
                     target_modules=targets, task_type='CAUSAL_LM')
    # DPO invocations get a FRESH process (no in-process SFT): building the probe
    # model here AND reloading in STAGE B would momentarily hold two 9GB models
    # in 16GB RAM -> swap thrash + MPS deadlock. So skip the probe when resuming.
    LORA_PARAMS = 6_624_768  # measured via get_peft_model for r=19, 8 depths
    print(f"trainable params: {LORA_PARAMS:,} (LoRA r={RANK}, depths={DEPTHS})", flush=True)
    model = None
    if not dpo_resumed:
        model = AutoModelForCausalLM.from_pretrained('google/gemma-4-E2B', dtype=torch.float16, low_cpu_mem_usage=False)
        model = get_peft_model(model, cfg)
        model.to(DEVICE)
    print(f"Device={DEVICE} MAXLEN={MAXLEN} SFT={SFT_STEPS} DPO={DPO_STEPS} accum={GRAD_ACCUM} "
          f"load={time.time()-t0:.0f}s", flush=True)

    # ---------- STAGE A: SFT ----------
    if not state.get('sft_complete', False):
        print('=' * 60, '\nSTAGE A: SFT (answer-only weighted CE, anchor 4x)', flush=True)
        ds = ECSFTDataset(DATA, tok, MAXLEN)
        loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=SFT_LR, weight_decay=0.0)
        remaining = max(SFT_STEPS - state.get('sft_step', 0), 1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining, eta_min=SFT_LR * 0.1)
        for _ in range(state.get('sft_step', 0) // GRAD_ACCUM):
            sched.step()
        model.train()
        losses, step = [], state.get('sft_step', 0)
        print(f"Starting SFT {step}/{SFT_STEPS} | rows={len(ds)}", flush=True)
        while step < SFT_STEPS:
            for batch in loader:
                if step >= SFT_STEPS:
                    break
                try:
                    ids = batch['input_ids'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)
                    w = batch['weights'].to(DEVICE)
                    logits = model(input_ids=ids).logits
                    loss = weighted_ce(logits, labels, w, tok.eos_token_id) / GRAD_ACCUM
                    if torch.isfinite(loss):
                        loss.backward()
                    if (step + 1) % GRAD_ACCUM == 0:
                        gn = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                        if torch.isfinite(gn):
                            opt.step()
                        opt.zero_grad(set_to_none=True)
                        sched.step()
                    losses.append(loss.item() * GRAD_ACCUM)
                    step += 1
                    state['sft_step'] = step
                    if step % 20 == 0:
                        gc.collect()
                        if DEVICE == 'mps':
                            torch.mps.empty_cache()
                    if step % LOG_EVERY == 0:
                        avg = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
                        print(f"  SFT {step}/{SFT_STEPS} loss={avg:.4f} "
                              f"({step/(time.time()-t0+1e-9):.2f} it/s)", flush=True)
                    if step % SAVE_EVERY == 0:
                        model.save_pretrained(f'{CKPT}/lora_baseline_sft')
                        save_state(state)
                except Exception as e:
                    print(f"  [SFT err] {e}", flush=True)
                    traceback.print_exc()
                    step += 1
        state['sft_complete'] = True
        save_state(state)
        model.save_pretrained(f'{CKPT}/lora_baseline_sft')
        print("STAGE A done -> lora_baseline_sft/ | exiting (run again for DPO "
              "in a fresh process: MPS allocator state from SFT deadlocks the "
              "double-forward DPO step)", flush=True)
        sys.exit(0)

    # ---------- STAGE B: DPO ----------
    if not state.get('dpo_complete', False):
        print('=' * 60, '\nSTAGE B: DPO (reference-free ranking, chosen vs base-wrong)', flush=True)
        # fresh process: free the startup probe model first, then load
        # base + SFT adapter (single model in memory at any time)
        from peft import PeftModel
        del model
        gc.collect()
        if DEVICE == 'mps':
            torch.mps.empty_cache()
            time.sleep(3)
        base = AutoModelForCausalLM.from_pretrained('google/gemma-4-E2B', dtype=torch.float16, low_cpu_mem_usage=False)
        model = PeftModel.from_pretrained(base, f'{CKPT}/lora_baseline_sft', is_trainable=True)
        model.to(DPO_DEVICE)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=DPO_LR, weight_decay=0.0)
        ds = ECDataset(DATA, tok, MAXLEN)
        loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
        model.train()
        losses, step = [], state.get('dpo_step', 0)
        print(f"Starting DPO {step}/{DPO_STEPS} | pairs={len(ds)}", flush=True)
        while step < DPO_STEPS:
            for batch in loader:
                if step >= DPO_STEPS:
                    break
                try:
                    c_ids = batch['chosen_ids'].to(DPO_DEVICE)
                    r_ids = batch['rejected_ids'].to(DPO_DEVICE)
                    lc = model(input_ids=c_ids).logits
                    lr_ = model(input_ids=r_ids).logits
                    lp_c = seq_logps(lc, c_ids)
                    lp_r = seq_logps(lr_, r_ids)
                    loss = -F.logsigmoid(DPO_BETA * (lp_c - lp_r)).mean() / GRAD_ACCUM
                    if torch.isfinite(loss):
                        loss.backward()
                    if (step + 1) % GRAD_ACCUM == 0:
                        gn = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                        if torch.isfinite(gn):
                            opt.step()
                        opt.zero_grad(set_to_none=True)
                    losses.append(loss.item() * GRAD_ACCUM)
                    step += 1
                    state['dpo_step'] = step
                    if step % 20 == 0:
                        gc.collect()
                        if DEVICE == 'mps':
                            torch.mps.empty_cache()
                    if step % LOG_EVERY == 0:
                        avg = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
                        print(f"  DPO {step}/{DPO_STEPS} loss={avg:.4f}", flush=True)
                    if step % SAVE_EVERY == 0:
                        model.save_pretrained(f'{CKPT}/lora_baseline_dpo')
                        save_state(state)
                except Exception as e:
                    print(f"  [DPO err] {e}", flush=True)
                    traceback.print_exc()
                    step += 1
        state['dpo_complete'] = True
        save_state(state)
        model.save_pretrained(f'{CKPT}/lora_baseline_dpo')
        print("STAGE B done -> lora_baseline_dpo/", flush=True)
        sys.exit(0)

    print(f"ALL DONE in {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == '__main__':
    main()
