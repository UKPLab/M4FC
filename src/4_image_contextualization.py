import os
import re
import json
import argparse
import pandas as pd
import concurrent.futures

from utils import dataset_loader, load_json_file
from utils_llm import get_commercial_model, prompt_commercial_model, json_to_xlsx, init_cols
from utils_openllms import get_open_model, prompt_open_model

# Unified prompts (match commercial script)
PMP_CONTEXTUALIZATION = """You are an AI assistant designed to help analyze information. Given the image and related \
article {evidence}, your task is to identify the following important information:

1. People: Who is shown in the image?
2. Things: Which animal, plant, building, or object are shown in the image?
3. Event: Which event is depicted in the image?
4. Date: When was the image taken?
5. Location: Where was the image taken?
6. Motivation: Why was the image taken?
7. Source: Who is the source of the image?

Please provide your answers in the following format in English:

"People": "[Answer]",
"Things": "[Answer]",
"Event": "[Answer]",
"Date": "[Answer]",
"Location": "[Answer]",
"Motivation": "[Answer]",
"Source": "[Answer]"


Please if there is no answer, just leave it blank, like empty string. Do not output anything else. The output must to be English.
"""

# Evidence and fake image mappings
EVI_FILE = "data/retrieval_results/evidence.json"
FAKE_IMAGE_MAP_FILE = "data/map_manipulated_original.json"

# Preload evidence and fake map
try:
    evi_set = json.load(open(EVI_FILE, encoding="utf-8"))
except Exception:
    evi_set = []

try:
    fake_image_map = load_json_file(FAKE_IMAGE_MAP_FILE)
except Exception:
    fake_image_map = {}


def resolve_image_path(rel_path: str) -> str:
    """Map dataset image_path to local data path, Windows-safe."""
    rel_path = (rel_path or "").strip().replace("\\", "/")
    path = os.path.normpath(os.path.join("data", *rel_path.split("/")))
    return path.replace("\\", "/")


def clean_bracketed_string(s: str) -> str:
    s = s.strip().replace("\u200b", "")
    s = re.sub(r'^[\'"\[\]（）{}<>“”‘’]*', '', s)
    s = re.sub(r'[\'"\[\]（）{}<>“”‘’]*[,，。；、\s]*$', '', s)
    return s.strip()


def get_all_evis(image_path: str) -> dict:
    all_evis = []
    selected_keys = ["title", "hostname", "description", "text", "date", "image_caption"]
    for data in evi_set:
        try:
            if image_path in data.get("image_path", ""):
                selected_data = {k: data[k] for k in selected_keys if k in data}
                result_str = json.dumps(selected_data, ensure_ascii=False)
                all_evis.append(result_str)
        except Exception:
            continue
    return {f"Evidence {idx}": evi for idx, evi in enumerate(all_evis)}


def process_row(idx, row, model_kind, model_obj, model_name):
    # Only compute when pred_all_replaced is missing
    if pd.isna(row.get("pred_all_replaced")):
        image_path = resolve_image_path(row["image_path"])  # standardize path
        evidence = get_all_evis(image_path)
        evidence_str = json.dumps(evidence, ensure_ascii=False)

        # Map manipulated image to original if needed
        if image_path in fake_image_map:
            image_path = fake_image_map[image_path]

        query = PMP_CONTEXTUALIZATION.format(evidence=evidence_str)
        try:
            if model_kind == "commercial":
                contextualization = prompt_commercial_model(model_obj, model_name, query, image_path, [])
            else:
                contextualization = prompt_open_model(model_obj, query, image_path, [])
            contextualization = (contextualization or "").strip()
        except Exception as e:
            contextualization = f"Error: {e}"

        return idx, {"pred_all_replaced": contextualization}

    # If already present, keep existing value
    return idx, {"pred_all_replaced": row.get("pred_all_replaced", "")}


def extract_fields(text: str) -> dict:
    fields = ["People", "Things", "Event", "Date", "Location", "Motivation", "Source"]
    results = {field: "" for field in fields}
    for field in fields:
        pattern = rf'"{field}":\s*(.*?)(?=\n"|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            value = match.group(1).strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            results[field] = value.strip()
    return results


def results_extraction(args):
    res_folder = os.path.join(f"res_{args.task}")
    result_file = os.path.join(res_folder, f"res_{args.task}_{args.model}.xlsx")
    df_input = pd.read_excel(result_file)

    for idx, row in df_input.iterrows():
        pred_all = row.get("pred_all_replaced")
        if pd.isna(pred_all):
            continue
        extracted = extract_fields(str(pred_all))
        for key, value in extracted.items():
            df_input.at[idx, f"pred_{key.lower()}_replaced"] = clean_bracketed_string(value)

    df_input.to_excel(result_file, index=False, engine='openpyxl')


def main(args):
    # Load dataset
    train_set, dev_set, test_set = dataset_loader(args.file, args.task)

    # Prepare result file
    res_folder = os.path.join(f"res_{args.task}")
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    result_file = os.path.join(res_folder, f"res_{args.task}_{args.model}.xlsx")
    if not os.path.exists(result_file):
        json_to_xlsx(test_set, result_file)

    df_input = pd.read_excel(result_file)

    # Ensure columns exist
    old_cols = [
        "pred_all", "pred_people", "pred_things", "pred_event",
        "pred_date", "pred_location", "pred_motivation", "pred_source",
    ]
    new_cols = [f"{item}_replaced" for item in old_cols]
    df_input = init_cols(df_input, new_cols)

    # Model selection: commercial vs open-source
    model_name = args.model.lower()
    if model_name in ["gpt4o", "gemini"]:
        model_kind = "commercial"
        model_obj = get_commercial_model(model_name)
    elif model_name in ["internvl", "llama", "qwenvl"]:
        model_kind = "open"
        model_obj = get_open_model(model_name)
    else:
        raise ValueError(
            f"Unsupported model: {args.model}. Choose from gpt4o, gemini, internvl, llama, qwenvl"
        )

    # Concurrency for commercial models, sequential for local models
    if model_kind == "commercial":
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(process_row, idx, row, model_kind, model_obj, model_name): idx
                for idx, row in df_input.iterrows()
                if idx <= args.nums
            }
            for future in concurrent.futures.as_completed(futures):
                idx, updated = future.result()
                for k, v in updated.items():
                    df_input.at[idx, k] = v
    else:
        for idx, row in df_input.iterrows():
            if idx > args.nums:
                break
            idx, updated = process_row(idx, row, model_kind, model_obj, model_name)
            for k, v in updated.items():
                df_input.at[idx, k] = v

    # Save contextualization and extract fields
    df_input.to_excel(result_file, index=False, engine='openpyxl')
    results_extraction(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/M4FC.json", help="file path")
    parser.add_argument("--model", type=str, default="internvl", help="gpt4o|gemini|internvl|llama|qwenvl")
    parser.add_argument("--task", type=str, default="image_contextualization")
    parser.add_argument("--nums", type=int, default=6000)
    args = parser.parse_args()
    main(args)