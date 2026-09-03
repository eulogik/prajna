#!/usr/bin/env python3
"""Quick demo: CRN v2 error correction on Gemma-4-E2B.

Usage:
    python inference.py                          # interactive mode
    python inference.py --prompt "..." --draft "..."  # single correction
"""
import argparse, torch
from crn_v2 import CRNv2


def main():
    p = argparse.ArgumentParser(description="CRN v2 error correction demo")
    p.add_argument("--prompt", type=str, help="The original question/prompt")
    p.add_argument("--draft", type=str, help="The draft answer to correct")
    p.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--checkpoint", type=str, default="checkpoints/crn_v2_dpo.pt")
    args = p.parse_args()

    print(f"Loading CRN v2 on {args.device}...")
    model = CRNv2(device=args.device)
    model.load(args.checkpoint)
    print("Ready.\n")

    if args.prompt and args.draft:
        result = model.correct(args.prompt, args.draft)
        print(f"Prompt:  {args.prompt}")
        print(f"Draft:   {args.draft}")
        print(f"Corrected: {result}")
        return

    print("Interactive mode. Type 'quit' to exit.\n")
    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        draft = input("Draft:  ").strip()
        if draft.lower() == 'quit':
            break

        base_out = model.base_generate(prompt)
        corrected = model.correct(prompt, draft)

        print(f"\nBase model says:  {base_out}")
        print(f"Draft answer:     {draft}")
        print(f"CRN v2 corrected: {corrected}\n")


if __name__ == '__main__':
    main()
