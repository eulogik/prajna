#!/usr/bin/env python3
"""CRN v2 trainer: frozen base + logit correction + KL preservation.

Mirrors train_prajna2b.py structure (ECSFTDataset → ECDataset, SFT → DPO)
but trains the CRN v2 LogitCorrection module instead.

Env vars:
  CRN_V2_SFT_STEPS  (default 2000)
  CRN_V2_DPO_STEPS  (default 500)
  CRN_V2_SFT_LR     (default 3e-4)
  CRN_V2_DPO_LR     (default 5e-6)
  CRN_V2_GRAD_ACCUM  (default 8)
  CRN_V2_KL_LAMBDA   (default 0.1)
  CRN_V2_CKPT_DIR    (default prajna/checkpoints)
  CRN_V2_MAXLEN      (default 96)
  CRN_V2_DEVICE      (default mps)
"""
import os, sys, json, time, gc, torch, torch.nn.functional as F, traceback
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
sys.path.insert(0, os.path.dirname(__file__))
from torch.utils.data import Dataset, DataLoader
from crn_v2 import CRNv2

# ── Hyperparams ──────────────────────────────────────────────────────────────
SFT_STEPS = int(os.environ.get('CRN_V2_SFT_STEPS', '2000'))
DPO_STEPS = int(os.environ.get('CRN_V2_DPO_STEPS', '500'))
SFT_LR = float(os.environ.get('CRN_V2_SFT_LR', '3e-4'))
DPO_LR = float(os.environ.get('CRN_V2_DPO_LR', '5e-6'))
GRAD_ACCUM = int(os.environ.get('CRN_V2_GRAD_ACCUM', '8'))
KL_LAMBDA = float(os.environ.get('CRN_V2_KL_LAMBDA', '0.1'))
CORRECTION_RANK = int(os.environ.get('CRN_V2_RANK', '128'))
CKPT = os.environ.get('CRN_V2_CKPT_DIR', 'prajna/checkpoints')
MAXLEN = int(os.environ.get('CRN_V2_MAXLEN', '96'))
STATE_PATH = os.path.join(CKPT, 'state_crn_v2.json')
DATA = 'prajna/data/error_correction_pairs_v2.json'
DEVICE = os.environ.get('CRN_V2_DEVICE', 'mps')
DPO_BETA = 0.1
MAX_GRAD_NORM = 1.0
LOG_EVERY = 20
SAVE_EVERY = 200


def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {}


def save_state(s):
    json.dump(s, open(STATE_PATH, 'w'))


class ECSFTDataset(Dataset):
    """Same as train_prajna2b.py:ECSFTDataset — SFT on chosen answers only."""
    def __init__(self, path, tok, ml):
        rows = json.load(open(path))
        self.data = []
        for s in rows:
            for variant in [s['chosen'], s.get('variant', s['chosen'])]:
                full = s['prompt'] + ': ' + variant + tok.eos_token
                enc = tok(full, truncation=True, max_length=ml, padding='max_length', return_tensors='pt')
                ids = enc['input_ids'].squeeze()
                labels = ids.clone()
                labels[enc['attention_mask'].squeeze() == 0] = -100
                prompt_len = len(tok(s['prompt'] + ': ', return_tensors='pt')['input_ids'][0])
                labels[:prompt_len] = -100
                # Anchor weighting: 4x at answer start, 0.5x at neighbors, 0.25x padding
                weights = torch.ones(ml, dtype=torch.float32) * 0.25
                weights[enc['attention_mask'].squeeze() == 1] = 1.0
                if prompt_len < ids.shape[0]:
                    labels[prompt_len - 1] = ids[prompt_len]
                    weights[prompt_len - 1] = 4.0
                    if prompt_len >= 3:
                        weights[prompt_len - 2] = 0.5
                        weights[prompt_len - 3] = 0.5
                self.data.append({
                    'input_ids': ids,
                    'labels': labels,
                    'pos_weights': weights,
                })

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


class ECDataset(Dataset):
    """Same as train_prajna2b.py:ECDataset — DPO on chosen vs rejected."""
    def __init__(self, path, tok, ml):
        rows = json.load(open(path))
        self.ch, self.rj = [], []
        for s in rows:
            c = tok(s['chosen'], truncation=True, max_length=ml, padding='max_length', return_tensors='pt')['input_ids'][0]
            r = tok(s['rejected'], truncation=True, max_length=ml, padding='max_length', return_tensors='pt')['input_ids'][0]
            self.ch.append(c)
            self.rj.append(r)

    def __len__(self): return len(self.ch)
    def __getitem__(self, i): return {'chosen_ids': self.ch[i], 'rejected_ids': self.rj[i]}


def weighted_ce(logits, labels, w, eos_id):
    """Same as train_prajna2b.py:weighted_ce — anchor 4x, eos 5x."""
    per_tok = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1), ignore_index=-100, reduction='none')
    per_tok = per_tok.reshape(labels.shape[0], -1)
    loss_w = w[:, 1:] if w.shape[1] == labels.shape[1] else w
    if eos_id is not None:
        eos_m = (labels[:, 1:] == eos_id).float()
        loss_w = loss_w * (1.0 + 4.0 * eos_m)
    eff = loss_w * (labels[:, 1:] != -100).float()
    return (per_tok * eff).sum() / eff.sum().clamp(min=1e-6)


def seq_logps(logits, ids):
    """Same as train_prajna2b.py:seq_logps."""
    logprobs = F.log_softmax(logits, dim=-1)
    tok_logprobs = torch.gather(logprobs[:, :-1], 2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = (ids[:, 1:] != 0).float()
    return (tok_logprobs * mask).sum(dim=-1)


def main():
    t0 = time.time()
    state = load_state()
    dpo_resumed = state.get('dpo_step', 0) > 0

    model = CRNv2(device=DEVICE, lambda_kl=KL_LAMBDA, correction_rank=CORRECTION_RANK)
    tok = model.tok
    params = model.get_params()

    print(f"Device={DEVICE} MAXLEN={MAXLEN} SFT={SFT_STEPS} DPO={DPO_STEPS} "
          f"accum={GRAD_ACCUM} kl={KL_LAMBDA} rank={CORRECTION_RANK} load={time.time()-t0:.0f}s", flush=True)

    # ── STAGE A: SFT ────────────────────────────────────────────────────────
    if not state.get('sft_complete', False):
        print('=' * 60, '\nSTAGE A: SFT (answer-only weighted CE, anchor 4x)', flush=True)
        ds = ECSFTDataset(DATA, tok, MAXLEN)
        loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
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
                    w = batch['pos_weights'].to(DEVICE)
                    out = model(input_ids=ids, labels=labels, eos_weight=5.0, pos_weights=w)
                    loss = out['loss'] / GRAD_ACCUM
                    if torch.isfinite(loss):
                        loss.backward()
                    if (step + 1) % GRAD_ACCUM == 0:
                        gn = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                        if torch.isfinite(gn):
                            opt.step()
                        opt.zero_grad(set_to_none=True)
                        sched.step()
                    losses.append(out['ce_loss'])
                    step += 1
                    state['sft_step'] = step
                    if step % LOG_EVERY == 0:
                        avg = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
                        print(f"  SFT {step}/{SFT_STEPS} loss={avg:.4f} (kl={out['kl_loss']:.4f})", flush=True)
                    if step % SAVE_EVERY == 0:
                        model.save(f'{CKPT}/crn_v2_sft.pt')
                        save_state(state)
                except Exception as e:
                    print(f"  [SFT err] {e}", flush=True)
                    traceback.print_exc()
                    step += 1
        model.save(f'{CKPT}/crn_v2_sft.pt')
        save_state(state)
        state['sft_complete'] = True
        save_state(state)
        print("STAGE A done -> crn_v2_sft.pt", flush=True)

    # ── STAGE B: DPO ────────────────────────────────────────────────────────
    if not state.get('dpo_complete', False):
        print('=' * 60, '\nSTAGE B: DPO (reference-free ranking, chosen vs base-wrong)', flush=True)
        # Load SFT checkpoint
        model.load(f'{CKPT}/crn_v2_sft.pt')
        params = model.get_params()
        ds = ECDataset(DATA, tok, MAXLEN)
        loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
        opt = torch.optim.AdamW(params, lr=DPO_LR, weight_decay=0.0)
        model.train()
        lc, lr_ = None, None
        losses, step = [], state.get('dpo_step', 0)
        print(f"Starting DPO {step}/{DPO_STEPS} | pairs={len(ds)}", flush=True)
        while step < DPO_STEPS:
            for batch in loader:
                if step >= DPO_STEPS:
                    break
                try:
                    c_ids = batch['chosen_ids'].to(DEVICE)
                    r_ids = batch['rejected_ids'].to(DEVICE)
                    lc = model(input_ids=c_ids)['logits']
                    lr_ = model(input_ids=r_ids)['logits']
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
                    if step % LOG_EVERY == 0:
                        avg = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
                        print(f"  DPO {step}/{DPO_STEPS} loss={avg:.4f}", flush=True)
                    if step % SAVE_EVERY == 0:
                        model.save(f'{CKPT}/crn_v2_dpo.pt')
                        save_state(state)
                except Exception as e:
                    print(f"  [DPO err] {e}", flush=True)
                    traceback.print_exc()
                    step += 1
        model.save(f'{CKPT}/crn_v2_dpo.pt')
        save_state(state)
        state['dpo_complete'] = True
        save_state(state)
        print("STAGE B done -> crn_v2_dpo.pt", flush=True)

    print("All stages complete.", flush=True)


if __name__ == '__main__':
    main()
