import argparse
import os
import concurrent.futures
import pandas as pd

from utils import dataset_loader
from utils_llm import get_commercial_model, prompt_commercial_model, json_to_xlsx, init_cols
from utils_openllms import get_open_model, prompt_open_model

PMP_ZERO = (
    "You are a helpful assistant designed to support fact-checking. "
    "Your task is to help initialize the prediction of the user's motivation/intention "
    "for sharing this image for further analysis. Clearly articulate it in a single, concise, "
    "and neutral sentence or short paragraph, starting with 'Motivation: To'"
)


def resolve_image_path(rel_path: str) -> str:
    """Match 2_claim_intent_prediction.py: map dataset image_path to local data path."""
    rel_path = (rel_path or "").strip().replace("\\", "/")
    path = os.path.normpath(os.path.join("data", *rel_path.split("/")))
    return path.replace("\\", "/")


def process_row(idx, row, model_kind, model_obj, model_name):
    image_path = resolve_image_path(row["image_path"])  # standardize path
    updated_values = {}
    if pd.isna(row.get("motivation_0")):
        if model_kind == "commercial":
            resp = prompt_commercial_model(model_obj, model_name, PMP_ZERO, image_path, [])
        else:
            resp = prompt_open_model(model_obj, PMP_ZERO, image_path, [])
        updated_values["motivation_0"] = (resp or "").lstrip("Motivation:").strip()
    return idx, updated_values


def main(args):
    # Load dataset and prepare result file (match commercial script conventions)
    train_set, dev_set, test_set = dataset_loader(args.file, args.task)

    res_folder = os.path.join(f"res_{args.task}")
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    result_file = os.path.join(res_folder, f"res_{args.task}_{args.model}.xlsx")
    if not os.path.exists(result_file):
        json_to_xlsx(test_set, result_file)

    df_input = pd.read_excel(result_file)
    df_input = init_cols(df_input, ["motivation_0"])  # ensure target column exists

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

    # Concurrency for commercial models (API-bound), sequential for local models
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

    df_input.to_excel(result_file, index=False, engine='openpyxl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/M4FC.json", help="file path")
    parser.add_argument(
        "--model", type=str, default="internvl", help="gpt4o|gemini|internvl|llama|qwenvl"
    )
    parser.add_argument("--task", type=str, default="claimant_motivation")
    parser.add_argument("--nums", type=int, default=2000)

    args = parser.parse_args()
    main(args)