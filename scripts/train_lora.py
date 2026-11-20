"""关卡 1：在轨迹数据上训练 LoRA（只算 agent 输出部分的损失）。

序列 = 用官方聊天模板渲染的完整对话（与推理时最后一步看到的上下文逐字相同）+ 最终输出。
损失只算每个 assistant 回合的正文和结束符；最后一回合模板里的空 <think></think> 不算。
需先停掉 vLLM 释放显存。保存的 LoRA 可直接用 vLLM --lora-modules 挂载。

用法：python scripts/train_lora.py --data artifacts/sft/ace20.jsonl --out artifacts/lora_ace20
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

TARGETS = ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj",
           "in_proj_qkv", "in_proj_z", "out_proj"]
AST = "<|im_start|>assistant\n"
END = "<|im_end|>"
THINK = "<think>\n\n</think>\n\n"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encode(tok, messages: list[dict], max_len: int, n_demo: int = 0):
    text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    # 找出每个 assistant 回合正文（含结束符）的字符区间
    spans, pos = [], 0
    while True:
        s = text.find(AST, pos)
        if s < 0:
            break
        b = s + len(AST)
        if text.startswith(THINK, b):
            b += len(THINK)
        e = text.find(END, b)
        e = len(text) if e < 0 else e + len(END)
        spans.append((b, e))
        pos = e
    spans = spans[n_demo:]  # 跳过官方提示词里的示范回合
    labels = [-100] * len(ids)
    j = 0
    for i, (a, b) in enumerate(offs):
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        if j < len(spans) and a >= spans[j][0] and b <= spans[j][1] and b > a:
            labels[i] = ids[i]
    if len(ids) > max_len:
        return None
    return torch.tensor(ids), torch.tensor(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=24576)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="从已有 LoRA 目录继续训练（不填 = 新建）")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("LoRA training requires a CUDA GPU")
    torch.manual_seed(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    data, skipped = [], 0
    for r in rows:
        x = encode(tok, r["messages"], args.max_len, r.get("n_demo_assistant", 0))
        if x is None or not (x[1][1:] != -100).any():
            skipped += 1
        else:
            data.append(x)
    if not data:
        ap.error("no usable training examples")
    n_tok = sum(len(x[0]) for x in data)
    n_lab = sum(int((x[1] != -100).sum()) for x in data)
    log(f"{len(data)} sequences ({skipped} skipped due to length or missing loss tokens), {n_tok} tokens, {n_lab} loss tokens")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if args.init:  # 增量蒸馏：在已有 LoRA 上继续训（框架 v1 的"加强已有"）
        from peft import PeftModel
        pm = PeftModel.from_pretrained(model, args.init, is_trainable=True)
    else:
        cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, target_modules=TARGETS,
                         exclude_modules=r".*(visual|mtp).*", task_type="CAUSAL_LM")
        pm = get_peft_model(model, cfg)
    pm.train()
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.epochs * math.ceil(len(data) / args.accum)  # 每轮末尾不满 accum 的也算一步
    warm = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(  # 超过总步数后学习率保持 0，不再回升
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    rng = random.Random(args.seed)
    step, t0, hist = 0, time.time(), []
    for ep in range(args.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        acc_loss, acc_n = 0.0, 0
        for k, i in enumerate(order):
            ids, lab = data[i]
            # 只在算损失的位置计算 logits：完整 logits（1.5 万 token × 24.8 万词表 fp32 ≈ 14GB）会显存溢出
            pos = (lab[1:] != -100).nonzero().flatten()  # 位置 p 的 logits 预测 token p+1
            out = pm(input_ids=ids[None].cuda(), logits_to_keep=pos.cuda())
            loss = torch.nn.functional.cross_entropy(out.logits[0].float(), ids[pos + 1].cuda())
            del out
            (loss / args.accum).backward()
            acc_loss += float(loss)
            acc_n += 1
            if acc_n == args.accum or k == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                hist.append(acc_loss / acc_n)
                log(f"epoch {ep} step {step}/{total} loss {acc_loss / acc_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
                acc_loss, acc_n = 0.0, 0
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(args.out)
    (Path(args.out) / "train_log.json").write_text(json.dumps(
        {"args": vars(args), "sequences": len(data), "skipped": skipped, "tokens": n_tok, "loss_tokens": n_lab,
         "loss_per_step": hist, "seconds": round(time.time() - t0)}, indent=1))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
