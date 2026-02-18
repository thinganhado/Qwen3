#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "polisher_prompt" / "region_forensics_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "polisher_prompt" / "region_forensics_user.txt"
DEFAULT_INPUT_QWEN3_VL_ROOT = Path("/scratch3/che489/Ha/interspeech/VLM/Qwen3-VL/outputs")
DEFAULT_GT_CSV = Path("/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/region_phone_table_grid.csv")
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


def _normalize_method(method: str) -> str:
    m = str(method or "").strip().lower()
    if m in {"", "grid"}:
        return "grid"
    return m


def _extract_explanation_and_answer(response_text: str) -> tuple[str, str]:
    explanation = ""
    answer = ""

    m_exp = re.search(r"<Explanation>\s*(.*?)\s*</Explanation>", response_text, flags=re.IGNORECASE | re.DOTALL)
    if m_exp:
        explanation = m_exp.group(1).strip()

    m_ans = re.search(r"<answer>\s*(.*?)\s*</answer>", response_text, flags=re.IGNORECASE | re.DOTALL)
    if m_ans:
        answer = m_ans.group(1).strip()

    return explanation, answer


def _build_metadata_xml(gt_row: dict | None, sample_id: str, region_id: int, method: str) -> tuple[str, dict]:
    if gt_row is None:
        meta = {
            "sample_id": sample_id,
            "method": method,
            "region_id": region_id,
            "time": "",
            "frequency": "",
            "phoneme": "",
        }
    else:
        meta = {
            "sample_id": sample_id,
            "method": method,
            "region_id": region_id,
            "time": str(gt_row.get("T", "")).strip(),
            "frequency": str(gt_row.get("F", "")).strip(),
            "phoneme": str(gt_row.get("P_type", "")).strip(),
        }

    metadata_xml = (
        "<META>\n"
        f"  <sample_id>{meta['sample_id']}</sample_id>\n"
        f"  <method>{meta['method']}</method>\n"
        f"  <region_id>{meta['region_id']}</region_id>\n"
        f"  <time>{meta['time']}</time>\n"
        f"  <frequency>{meta['frequency']}</frequency>\n"
        f"  <phoneme>{meta['phoneme']}</phoneme>\n"
        "</META>"
    )
    return metadata_xml, meta


def _load_gt_index(gt_csv_path: Path) -> tuple[dict, dict]:
    by_full_key = {}
    by_sample_region = {}
    if not gt_csv_path.exists():
        raise FileNotFoundError(f"--gt-csv does not exist: {gt_csv_path}")

    with gt_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            method = _normalize_method(row.get("method", "grid"))
            rid_raw = str(row.get("region_id", "")).strip()
            if not sample_id or not rid_raw:
                continue
            try:
                rid = int(rid_raw)
            except Exception:
                continue

            by_full_key[(sample_id, method, rid)] = row
            by_sample_region[(sample_id, rid)] = row

    return by_full_key, by_sample_region


def _read_input_items(args: argparse.Namespace):
    items = []
    input_root = Path(args.input_qwen3_vl_root).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"--input-qwen3-vl-root does not exist: {input_root}")

    gt_csv_path = Path(args.gt_csv).expanduser().resolve()
    gt_by_full_key, gt_by_sample_region = _load_gt_index(gt_csv_path)

    # Layout: <root>/<method>/<sample_id>/json
    for json_path in sorted(input_root.glob("*/*/json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid grouped JSON file: {json_path}. Error: {e}") from e

        sample_id = str(payload.get("sample_id", json_path.parent.name)).strip()
        method_name = _normalize_method(json_path.parent.parent.name)
        regions = payload.get("regions", [])
        if not isinstance(regions, list):
            continue

        for region in regions:
            if not isinstance(region, dict):
                continue

            response_text = str(region.get("response", "")).strip()
            if not response_text:
                continue

            try:
                region_id = int(region.get("region_id", 0))
            except Exception:
                continue

            gt_row = gt_by_full_key.get((sample_id, method_name, region_id))
            if gt_row is None:
                gt_row = gt_by_sample_region.get((sample_id, region_id))

            metadata_xml, _ = _build_metadata_xml(
                gt_row=gt_row,
                sample_id=sample_id,
                region_id=region_id,
                method=method_name,
            )

            explanation, answer = _extract_explanation_and_answer(response_text)
            if explanation:
                description = (
                    f"Region ID: {region_id}\n"
                    f"Explanation: {explanation}\n"
                    f"Answer: {answer}"
                )
            else:
                description = (
                    f"Region ID: {region_id}\n"
                    f"Response: {response_text}\n"
                    f"Answer: {answer}"
                )

            items.append(
                {
                    "sample_id": sample_id,
                    "region_id": region_id,
                    "description": description,
                    "metadata_xml": metadata_xml,
                    "source_response": response_text,
                }
            )

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


def _build_user_prompt(template: str, item: dict) -> str:
    prompt = template.format(
        description=str(item.get("description", "")),
        metadata_xml=str(item.get("metadata_xml", "")),
    ).strip()
    if not prompt:
        raise ValueError("User template rendered empty text.")
    return prompt


def build_messages(args: argparse.Namespace, item: dict):
    system_prompt = _resolve_system_prompt(args)
    user_template = _resolve_user_template(args)
    user_prompt = _build_user_prompt(user_template, item)

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
    parser.add_argument("--gt-csv", default=str(DEFAULT_GT_CSV), help="GT CSV with sample_id/method/region_id/T/F/P_type.")
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
                    "description": item["description"],
                    "metadata_xml": item["metadata_xml"],
                    "source_response": item["source_response"],
                    "prompt": user_prompt,
                    "response": output_text,
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


