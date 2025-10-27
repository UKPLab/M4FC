import argparse
import os
import concurrent.futures
import json
import pandas as pd

from utils import dataset_loader
from utils_llm import get_commercial_model, prompt_commercial_model, json_to_xlsx, init_cols
from utils_openllms import get_open_model, prompt_open_model

PMP_VERACITY = (
    """
You are an AI assistant designed to evaluate the veracity of multimodal claims. Given the claim and the related image, your task is to analyze the provided information and predict the more likely veracity: True or False. Please do not refuse or respond with "not enough information." Your prediction will assist professional fact-checkers in further analysis.

Please provide your prediction as 'Veracity: True' or 'Veracity: False'.

In one or more paragraphs, output your reasoning steps. In the separate final line, output your prediction. (Please don't output anything else except the Veracity Predicion )

Claim: {CLAIM}

Let's think step by step.
"""
)

PMP_VERACITY_EVI = (
    """
You are an AI assistant designed to evaluate the veracity of multimodal claims. Given the claim and the related image, your task is to analyze the provided information and predict the more likely veracity: True or False. Please do not refuse or respond with "not enough information." Your prediction will assist professional fact-checkers in further analysis.

Please provide your prediction as 'Veracity: True' or 'Veracity: False'.

In one or more paragraphs, output your reasoning steps. In the separate final line, output your prediction. (Please don't output anything else except the Veracity Predicion )

Claim: {CLAIM}

Retrieved Evidences:\n {EVIDENCE}

Let's think step by step.
"""
)

# Attempt to load evidence set; fallback gracefully if file missing
EVI_FILE = "data/retrieval_results/evidence.json"
try:
    evi_set = json.load(open(EVI_FILE, encoding="utf-8"))
except Exception:
    evi_set = []


def resolve_image_path(rel_path: str) -> str:
    """Match 6_verdict_prediction.py: map dataset image_path to local data path."""
    rel_path = (rel_path or "").strip().replace("\\", "/")
    path = os.path.normpath(os.path.join("data", *rel_path.split("/")))
    return path.replace("\\", "/")


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
    updated_values = {}
    claim_text = row.get("claim", "")
    image_path = resolve_image_path(row.get("image_path", ""))

    # veracity_0: claim + image
    if pd.isna(row.get("veracity_0")):
        prompt = PMP_VERACITY.format(CLAIM=claim_text)
        if model_kind == "commercial":
            resp = prompt_commercial_model(model_obj, model_name, prompt, image_path, [])
        else:
            resp = prompt_open_model(model_obj, prompt, image_path, [])
        updated_values["veracity_0"] = (resp or "").lstrip("Veracity:").strip()

    # veracity_evi: claim + image + retrieved evidences
    if pd.isna(row.get("veracity_evi")):
        total_evis = get_all_evis(image_path)
        prompt_evi = PMP_VERACITY_EVI.format(CLAIM=claim_text, EVIDENCE=total_evis)
        if model_kind == "commercial":
            resp_evi = prompt_commercial_model(model_obj, model_name, prompt_evi, image_path, [])
        else:
            resp_evi = prompt_open_model(model_obj, prompt_evi, image_path, [])
        updated_values["veracity_evi"] = (resp_evi or "").lstrip("Veracity:").strip()

    return idx, updated_values


def main(args):
    # Load dataset
    train_set, dev_set, test_set = dataset_loader(args.file, args.task)
    # Prepare output file
    res_folder = os.path.join(f"res_{args.task}")
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    result_file = os.path.join(res_folder, f"res_{args.task}_{args.model}.xlsx")

    # if not os.path.exists(result_file):
        # Use full test_set for consistency with other open scripts
    json_to_xlsx(test_set, result_file)

    # Read and ensure columns
    df_input = pd.read_excel(result_file)
    df_input = init_cols(df_input, ["veracity_0", "veracity_evi"])  

    # Model selection
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

    # Concurrency for commercial models; sequential for local open models
    if model_kind == "commercial":
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(process_row, idx, row, model_kind, model_obj, model_name): idx
                for idx, row in df_input.iterrows()
                if idx <= args.nums
            }
            for future in concurrent.futures.as_completed(futures):
                idx, updated_values = future.result()
                for key, value in updated_values.items():
                    df_input.at[idx, key] = value
    else:
        for idx, row in df_input.iterrows():
            if idx > args.nums:
                break
            idx, updated_values = process_row(idx, row, model_kind, model_obj, model_name)
            for key, value in updated_values.items():
                df_input.at[idx, key] = value

    # Save results
    df_input.to_excel(result_file, index=False, engine='openpyxl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/M4FC.json", help="file path")
    parser.add_argument("--model", type=str, default="gemini", help="gpt4o|gemini|internvl|llama|qwenvl")
    parser.add_argument("--task", type=str, default="verdict_prediction")
    parser.add_argument("--nums", type=int, default=10)

    args = parser.parse_args()
    main(args)