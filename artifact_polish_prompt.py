#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "polisher_prompt" / "region_forensics_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "polisher_prompt" / "region_forensics_user.txt"
DEFAULT_INPUT_JSONL = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final__En/Qwen3-VL-30B-A3B-Thinking_/qwen_region_outputs_shard_0_of_4.jsonl"
DEFAULT_GT_CSV = Path("/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/region_phone_table_grid.csv")
DEFAULT_OUTPUT_DIR = Path("/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En/polisher/")
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/LLM/Qwen3-4B-Instruct-2507"


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


def _extract_explanation_block(response_text: str) -> str:
    m_exp = re.search(
        r"<Explanation>\s*(.*?)\s*(?:</Explanation>|</think>)",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_exp:
        inner = m_exp.group(1).strip()
        return f"<Explanation>{inner}</Explanation>"
    return ""


def _extract_json_block(response_text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    m_obj = re.search(r"(\{.*\})", response_text, flags=re.DOTALL)
    if not m_obj:
        return {}
    try:
        obj = json.loads(m_obj.group(1))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _parse_model_output(response_text: str, expected_region_id: int) -> dict:
    explanation = _extract_explanation_block(response_text)
    parsed = _extract_json_block(response_text)

    def _pick(*keys: str, default: str = "ambiguous") -> str:
        for k in keys:
            v = parsed.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return default

    region_id_val = _pick("region_id", default=str(expected_region_id))
    structured = {
        "region_id": str(region_id_val),
        "time": _pick("time"),
        "frequency": _pick("frequency"),
        "phonetic": _pick("phonetic", "phoneme"),
    }
    return {
        "output_explanation": explanation,
        "output_structured": structured,
        "output_raw": response_text,
    }


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


def _read_input_items_from_jsonl(args: argparse.Namespace):
    items = []
    jsonl_path = Path(args.input_jsonl).expanduser().resolve()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"--input-jsonl does not exist: {jsonl_path}")

    gt_csv_path = Path(args.gt_csv).expanduser().resolve()
    gt_by_full_key, gt_by_sample_region = _load_gt_index(gt_csv_path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            sample_id = str(payload.get("sample_id", "")).strip()
            response_text = str(payload.get("response", "")).strip()
            if not sample_id or not response_text:
                continue

            try:
                region_id = int(payload.get("region_id", ""))
            except Exception:
                continue

            method_name = _normalize_method(payload.get("crop_method", "grid"))
            gt_row = gt_by_full_key.get((sample_id, method_name, region_id))
            if gt_row is None:
                gt_row = gt_by_sample_region.get((sample_id, region_id))
            if gt_row is None:
                continue

            try:
                region_id_csv = int(str(gt_row.get("region_id", "")).strip())
            except Exception:
                continue

            explanation_block = _extract_explanation_block(response_text)
            if not explanation_block:
                continue

            description = f"Region ID: {region_id_csv}\n{explanation_block}"
            items.append(
                {
                    "sample_id": sample_id,
                    "region_id": region_id_csv,
                    "description": description,
                }
            )

    if args.max_items is not None:
        items = items[: args.max_items]
    if not items:
        raise ValueError("No input items found from --input-jsonl.")

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
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Region-output JSONL input path.",
    )
    parser.add_argument("--gt-csv", default=str(DEFAULT_GT_CSV), help="GT CSV with sample_id/method/region_id.")
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
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=["transformers", "vllm"],
        help="Inference backend. Use vllm for high-throughput generation.",
    )
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
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="vLLM tensor parallel size.")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization fraction.")
    parser.add_argument("--vllm-enforce-eager", action="store_true", help="Enable vLLM eager mode.")

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


def _normalize_dtype_name(dtype_str: str) -> str:
    key = str(dtype_str).strip().lower()
    aliases = {
        "fp16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
    }
    return aliases.get(key, key)


def _import_vllm_deps():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise ImportError("vllm is required for --backend vllm.") from e
    return LLM, SamplingParams


def _infer_tp_size_from_env() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        if parts:
            return len(parts)
    return 1


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


def _build_vllm_sampling_params(args, SamplingParams):
    sample_flag = bool(args.do_sample and args.temperature > 0.0)
    temperature = args.temperature if sample_flag else 0.0
    top_p = args.top_p if sample_flag else 1.0
    top_k = args.top_k if sample_flag else -1
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[],
    )


def _generate_batch_vllm(llm, tokenizer, batch_messages, sampling_params):
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in batch_messages
    ]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    texts = []
    for out in outputs:
        if getattr(out, "outputs", None):
            texts.append(out.outputs[0].text.strip())
        else:
            texts.append("")
    return texts


def _write_sample_grouped_json(output_dir: Path, records_by_sample: dict):
    # Keep function name for compatibility, but write per-region files:
    # <output_dir>/<sample_id>/<region_id>.json
    for sample_id, records in records_by_sample.items():
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        by_region = {}
        for rec in records:
            rid = rec.get("region_id")
            if rid is None:
                continue
            by_region[int(rid)] = rec

        for rid in sorted(by_region.keys()):
            out_file = sample_dir / f"{rid}.json"
            out_file.write_text(json.dumps(by_region[rid], ensure_ascii=False, indent=2), encoding="utf-8")


def _load_existing_records_by_sample(output_dir: Path) -> dict:
    records_by_sample = defaultdict(list)
    if not output_dir.exists():
        return records_by_sample

    # New structure: <output_dir>/<sample_id>/<region_id>.json
    for p in output_dir.glob("*/*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sample_id = str(obj.get("sample_id", p.parent.name))
        try:
            rid = int(obj.get("region_id", p.stem))
        except Exception:
            continue

        raw = str(obj.get("output_raw", obj.get("response", ""))).strip()
        parsed = _parse_model_output(raw, rid)
        normalized = {
            "sample_id": sample_id,
            "region_id": rid,
            "prompt": str(obj.get("prompt", "")),
            "response": raw,
            "output_raw": parsed["output_raw"],
            "output_explanation": parsed["output_explanation"],
            "output_structured": parsed["output_structured"],
        }
        records_by_sample[str(sample_id)].append(normalized)

    # Backward compatibility: old grouped structure <output_dir>/<sample_id>/json
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
            if not isinstance(rec, dict) or "region_id" not in rec:
                continue
            try:
                rid = int(rec.get("region_id"))
            except Exception:
                continue
            raw = str(rec.get("output_raw", rec.get("response", ""))).strip()
            parsed = _parse_model_output(raw, rid)
            normalized = {
                "sample_id": str(sample_id),
                "region_id": rid,
                "prompt": str(rec.get("prompt", "")),
                "response": raw,
                "output_raw": parsed["output_raw"],
                "output_explanation": parsed["output_explanation"],
                "output_structured": parsed["output_structured"],
            }
            records_by_sample[str(sample_id)].append(normalized)
    return records_by_sample


def _resume_scan_dirs(output_dir: Path, num_shards: int) -> list[Path]:
    """
    Return directories to scan for existing outputs in resume mode.
    If output_dir is a shard path like shard_<id>_of_<N>, scan all sibling shards.
    """
    dirs = []

    if output_dir.exists() and output_dir.is_dir():
        dirs.append(output_dir)

    m = re.match(r"^shard_\d+_of_(\d+)$", output_dir.name)
    if m:
        shard_count = int(m.group(1))
        parent = output_dir.parent
        for p in sorted(parent.glob(f"shard_*_of_{shard_count}")):
            if p.is_dir() and p not in dirs:
                dirs.append(p)
        return dirs

    if num_shards > 1 and output_dir.exists() and output_dir.is_dir():
        for p in sorted(output_dir.glob(f"shard_*_of_{num_shards}")):
            if p.is_dir() and p not in dirs:
                dirs.append(p)

    return dirs


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
    items = _read_input_items_from_jsonl(args)
    output_root = Path(args.output_dir).expanduser().resolve()

    existing_records = defaultdict(list)
    if not args.overwrite:
        scan_dirs = _resume_scan_dirs(output_root, args.num_shards)
        if len(scan_dirs) == 0:
            scan_dirs = [output_root]

        print(f"[resume] scanning_existing_dirs={len(scan_dirs)}")
        for d in scan_dirs:
            loaded = _load_existing_records_by_sample(d)
            for sid, recs in loaded.items():
                existing_records[sid].extend(recs)

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

    model = None
    llm = None
    sampling_params = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.backend == "transformers":
        torch_dtype = _resolve_torch_dtype(args.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            trust_remote_code=True,
        ).eval()
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    elif args.backend == "vllm":
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        LLM, SamplingParams = _import_vllm_deps()
        tp_size = args.tensor_parallel_size
        if tp_size is None:
            tp_size = _infer_tp_size_from_env()
        llm = LLM(
            model=args.model_id,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            dtype=_normalize_dtype_name(args.dtype),
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            enforce_eager=args.vllm_enforce_eager,
            seed=0,
        )
        sampling_params = _build_vllm_sampling_params(args, SamplingParams)
    else:
        raise ValueError(f"Unsupported --backend: {args.backend}")

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

            if args.backend == "vllm":
                batch_outputs = _generate_batch_vllm(
                    llm=llm,
                    tokenizer=tokenizer,
                    batch_messages=batch_messages,
                    sampling_params=sampling_params,
                )
            else:
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
                parsed = _parse_model_output(output_text, int(item["region_id"]))
                record = {
                    "sample_id": item["sample_id"],
                    "region_id": item["region_id"],
                    "prompt": user_prompt,
                    "response": output_text,
                    "output_raw": parsed["output_raw"],
                    "output_explanation": parsed["output_explanation"],
                    "output_structured": parsed["output_structured"],
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



