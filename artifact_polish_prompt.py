#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from string import Formatter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "prompts" / "region_forensics_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "prompts" / "region_forensics_user.txt"
DEFAULT_INPUT_QWEN3_VL_ROOT = Path("/scratch3/che489/Ha/interspeech/localization/qwen3_vlm")
DEFAULT_OUTPUT_DIR = Path("/scratch3/che489/Ha/interspeech/localization/qwen3_polished")
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/LLM/Qwen3-30B-A3B-Instruct-2507/"


def _load_text_file(path: Path, field_name: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} file does not exist: {resolved}")
    return resolved.read_text(encoding="utf-8").strip()


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    if args.system_file is None:
        return _load_text_file(DEFAULT_SYSTEM_FILE, "--system-file")
    return _load_text_file(Path(args.system_file), "--system-file")


def _resolve_user_template(args: argparse.Namespace) -> str:
    if args.user_template_file is None:
        return _load_text_file(DEFAULT_USER_TEMPLATE_FILE, "--user-template-file")
    return _load_text_file(Path(args.user_template_file), "--user-template-file")


def _read_input_items(args: argparse.Namespace):
    items = []
    if args.input_qwen3_vl_root:
        input_root = Path(args.input_qwen3_vl_root).expanduser().resolve()
        if not input_root.exists():
            raise FileNotFoundError(f"--input-qwen3-vl-root does not exist: {input_root}")

        # Layout: <root>/<method>/<sample_id>/json
        for json_path in sorted(input_root.glob("*/*/json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                raise ValueError(f"Invalid grouped JSON file: {json_path}. Error: {e}") from e

            sample_id = str(payload.get("sample_id", json_path.parent.name))
            method_name = json_path.parent.parent.name
            regions = payload.get("regions", [])
            if not isinstance(regions, list):
                continue

            for region in regions:
                if not isinstance(region, dict):
                    continue

                response_text = str(region.get("response", "")).strip()
                if not response_text:
                    continue

                region_id = region.get("region_id", 0)
                meta = {
                    "sample_id": sample_id,
                    "region_id": region_id,
                    "crop_method": str(region.get("crop_method", method_name)),
                    "time": str(region.get("time", "")),
                    "frequency": str(region.get("frequency", "")),
                    "phoneme": str(region.get("phoneme", "")),
                    "feature": str(region.get("feature", "")),
                }
                metadata_line = (
                    f'time="{meta["time"]}" '
                    f'frequency="{meta["frequency"]}" '
                    f'phoneme="{meta["phoneme"]}" '
                    f'feature="{meta["feature"]}"'
                )
                metadata_xml = (
                    "<META>\n"
                    f"  <time>{meta['time']}</time>\n"
                    f"  <frequency>{meta['frequency']}</frequency>\n"
                    f"  <phoneme>{meta['phoneme']}</phoneme>\n"
                    f"  <feature>{meta['feature']}</feature>\n"
                    "</META>"
                )
                items.append(
                    {
                        "sample_id": sample_id,
                        "region_id": region_id,
                        "method": method_name,
                        "response": response_text,
                        "description": response_text,
                        "artifact_description": response_text,
                        "metadata": metadata_line,
                        "metadata_xml": metadata_xml,
                        "metadata_obj": meta,
                        "metadata_json": json.dumps(meta, ensure_ascii=False),
                        "input_text": response_text,
                    }
                )

    elif args.input_jsonl:
        input_path = Path(args.input_jsonl).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"--input-jsonl does not exist: {input_path}")
        with input_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {lineno}: {e}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"Line {lineno} must be a JSON object")
                items.append(obj)

    elif args.input_csv:
        input_path = Path(args.input_csv).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"--input-csv does not exist: {input_path}")
        with input_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(dict(row))

    elif args.prompt:
        items = [{"sample_id": "single", "region_id": 0, "input_text": args.prompt}]
    else:
        raise ValueError("Provide one of --input-qwen3-vl-root, --input-jsonl, --input-csv, or --prompt.")

    if args.max_items is not None:
        items = items[: args.max_items]

    if not items:
        raise ValueError("No input items found.")

    normalized = []
    for idx, item in enumerate(items):
        sample_id = str(item.get("sample_id", "sample"))
        raw_region_id = item.get("region_id", idx)
        try:
            region_id = int(raw_region_id)
        except Exception:
            region_id = idx
        item2 = dict(item)
        item2["sample_id"] = sample_id
        item2["region_id"] = region_id
        normalized.append(item2)
    return normalized


def _build_user_prompt(template: str, item: dict, user_field: str) -> str:
    # Support both named placeholders (e.g., {response}) and positional placeholders ({}).
    has_positional = False
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name == "":
            has_positional = True
            break

    if has_positional:
        prompt = template.format(
            str(item.get("artifact_description", item.get(user_field, ""))),
            str(item.get("metadata_json", "")),
        ).strip()
    else:
        # Keep templating forgiving so new columns can be gradually added without breaking runs.
        field_map = defaultdict(str, item)
        if not field_map["description"]:
            field_map["description"] = (
                field_map["artifact_description"]
                or field_map["response"]
                or field_map[user_field]
            )
        prompt = template.format_map(field_map).strip()

    if prompt:
        return prompt
    return str(item.get(user_field, "")).strip()


def build_messages(args: argparse.Namespace, item: dict):
    system_prompt = _resolve_system_prompt(args)
    user_template = _resolve_user_template(args)
    user_prompt = _build_user_prompt(user_template, item, args.user_field)
    if not user_prompt:
        raise ValueError(
            f"Empty user prompt for sample_id={item['sample_id']} region_id={item['region_id']}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages, user_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local HF Qwen3 text prompt runner for region-level analysis."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HF model id or local model path.",
    )

    parser.add_argument(
        "--input-qwen3-vl-root",
        default=str(DEFAULT_INPUT_QWEN3_VL_ROOT),
        help="Root folder containing Qwen3-VL grouped outputs: <root>/<method>/<sample_id>/json",
    )
    parser.add_argument("--input-jsonl", default=None, help="Input JSONL file path.")
    parser.add_argument("--input-csv", default=None, help="Input CSV file path.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single prompt input (quick test path).",
    )
    parser.add_argument(
        "--user-field",
        default="input_text",
        help="Field name used when template renders empty.",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for discovered items.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split discovered items across N shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards).")

    parser.add_argument(
        "--system-file",
        default=None,
        help=("Path to system prompt txt. Default: " f"{DEFAULT_SYSTEM_FILE.as_posix()}"),
    )
    parser.add_argument(
        "--user-template-file",
        default=None,
        help=("Path to user template txt. Default: " f"{DEFAULT_USER_TEMPLATE_FILE.as_posix()}"),
    )

    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--dtype", default="auto", help="Model dtype: auto, float16, bfloat16, float32.")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of items per forward pass.")
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic sampling. If false, decoding is deterministic.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)

    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Per-sample output root.")
    parser.add_argument("--output-file", default=None, help="Single-item output file.")
    parser.add_argument("--output-jsonl", default=None, help="Optional flat batch output jsonl file.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Regenerate outputs even if per-sample region records already exist.",
    )
    parser.add_argument("--print-messages", action="store_true", help="Print built messages before generation.")
    return parser.parse_args()


def _resolve_torch_dtype(dtype_str: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported --dtype: {dtype_str}. Use one of: {list(mapping.keys())}")
    return mapping[dtype_str]


def _generate_batch(
    model,
    tokenizer,
    batch_messages,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    top_k,
):
    chat_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in batch_messages
    ]

    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True).to(model.device)
    input_lengths = inputs["attention_mask"].sum(dim=1)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
        generate_kwargs["top_k"] = top_k

    generated_ids = model.generate(**inputs, **generate_kwargs)

    outputs = []
    for i in range(generated_ids.shape[0]):
        output_ids = generated_ids[i][int(input_lengths[i]) :]
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        outputs.append(text)
    return outputs


def _write_sample_grouped_json(output_dir: Path, records_by_sample: dict):
    for sample_id, records in records_by_sample.items():
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        by_region = {}
        for rec in records:
            rid = rec.get("region_id")
            if rid is None:
                continue
            by_region[int(rid)] = rec
        records_sorted = [by_region[rid] for rid in sorted(by_region.keys())]
        payload = {
            "sample_id": sample_id,
            "num_regions": len(records_sorted),
            "regions": records_sorted,
        }

        out_file = sample_dir / "json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_existing_records_by_sample(output_dir: Path) -> dict:
    records_by_sample = defaultdict(list)
    if not output_dir.exists():
        return records_by_sample

    for p in output_dir.glob("*/json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sample_id = obj.get("sample_id")
        regions = obj.get("regions")
        if not sample_id or not isinstance(regions, list):
            continue
        for rec in regions:
            if isinstance(rec, dict) and "region_id" in rec:
                records_by_sample[str(sample_id)].append(rec)
    return records_by_sample


def _existing_done_keys(records_by_sample: dict) -> set:
    done = set()
    for sample_id, records in records_by_sample.items():
        for rec in records:
            rid = rec.get("region_id")
            if rid is None:
                continue
            done.add((str(sample_id), int(rid)))
    return done


def main():
    args = parse_args()
    items = _read_input_items(args)
    output_root = Path(args.output_dir).expanduser().resolve()

    existing_records = defaultdict(list)
    if not args.overwrite:
        existing_records = _load_existing_records_by_sample(output_root)
        done_keys = _existing_done_keys(existing_records)
        before = len(items)
        items = [
            it for it in items if (str(it["sample_id"]), int(it["region_id"])) not in done_keys
        ]
        skipped = before - len(items)
        if skipped > 0:
            print(f"[resume] skipped_existing_regions={skipped}")
        if len(items) == 0:
            print("[resume] no pending regions; nothing to generate.")
            return

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
        print(f"[shard] shard_id={args.shard_id}/{args.num_shards} items={len(items)}")
        if not items:
            raise ValueError("No items assigned to this shard.")

    if len(items) > 1 and args.output_file:
        raise ValueError("--output-file is only for single item. Use --output-dir for grouped outputs.")

    torch_dtype = _resolve_torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    jsonl_fp = None
    if args.output_jsonl:
        out_path = Path(args.output_jsonl).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if args.overwrite else "a"
        jsonl_fp = out_path.open(mode, encoding="utf-8")

    records_by_sample = defaultdict(list)
    if not args.overwrite:
        for sid, recs in existing_records.items():
            records_by_sample[sid].extend(recs)

    try:
        for batch_start in range(0, len(items), args.batch_size):
            batch_items = items[batch_start : batch_start + args.batch_size]
            batch_built = [build_messages(args, item) for item in batch_items]
            batch_messages = [x[0] for x in batch_built]
            batch_user_prompts = [x[1] for x in batch_built]

            if args.print_messages:
                for msg in batch_messages:
                    print(msg)

            try:
                batch_outputs = _generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    batch_messages=batch_messages,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
            except Exception as e:
                print(f"[batch-fallback] batch_size={len(batch_messages)} reason={e}")
                batch_outputs = []
                for messages in batch_messages:
                    batch_outputs.extend(
                        _generate_batch(
                            model=model,
                            tokenizer=tokenizer,
                            batch_messages=[messages],
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                    )

            for i, (item, user_prompt, output_text) in enumerate(
                zip(batch_items, batch_user_prompts, batch_outputs),
                start=1,
            ):
                idx = batch_start + i
                record = {
                    "sample_id": item["sample_id"],
                    "region_id": item["region_id"],
                    "prompt": user_prompt,
                    "response": output_text,
                    "input": item,
                }
                records_by_sample[record["sample_id"]].append(record)

                print(f"[{idx}/{len(items)}] {record['sample_id']}__r{record['region_id']}")
                print(output_text)

                if jsonl_fp is not None:
                    jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

                if len(items) == 1 and args.output_file:
                    out_file = Path(args.output_file).expanduser().resolve()
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_text(output_text, encoding="utf-8")
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()

    _write_sample_grouped_json(output_root, records_by_sample)


if __name__ == "__main__":
    main()
