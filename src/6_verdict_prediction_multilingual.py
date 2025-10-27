import argparse, os, concurrent.futures, pandas as pd, json
from collections import Counter
from utils import dataset_loader
from utils_llm import *
from utils_openllms import *

PMP_VERACITY = """
You are an AI assistant designed to evaluate the veracity of multimodal claims. Given the claim and the related image, your task is to analyze the provided information and predict the more likely veracity: True or False. Please do not refuse or respond with "not enough information." Your prediction will assist professional fact-checkers in further analysis.

Please provide your prediction as 'Veracity: True' or 'Veracity: False'.

In one or more paragraphs, output your reasoning steps. In the separate final line, output your prediction. (Please don't output anything else except the Veracity Predicion )

Claim: {CLAIM}

Let's think step by step.
"""

PMP_VERACITY_EVI = """
You are an AI assistant designed to evaluate the veracity of multimodal claims. Given the claim, the related image and retrieved evidence, your task is to analyze the provided information and predict the more likely veracity: True or False. Please do not refuse or respond with "not enough information." Your prediction will assist professional fact-checkers in further analysis.

Please provide your prediction as 'Veracity: True' or 'Veracity: False'.

In one or more paragraphs, output your reasoning steps. In the separate final line, output your prediction. (Please don't output anything else except the Veracity Predicion )

Claim: {CLAIM}

Retrieved Evidences:\n {EVIDENCE}

Let's think step by step.
"""

evi_file = "data/retrieval_results/evidence.json"
evi_set = json.load(open(evi_file, 'r', encoding='utf-8'))
image_evi_map_sorted = json.load(open("data/image_evi_map_sorted.json", 'r', encoding='utf-8'))

def get_all_evis(image_path):
    all_evis = []
    sel = ["title","hostname","description","text","date","image_caption"]
    for data in evi_set:
        if image_path in data.get("image_path", ""):
            sd = {k: data[k] for k in sel if k in data}
            all_evis.append(json.dumps(sd, ensure_ascii=False))
    return {f"Evidence {i}": e for i, e in enumerate(all_evis)}

def check_contains(ss, ll):
    return any(item in ss for item in ll)

def resolve_image_path(rel_path: str) -> str:
    rel_path = (str(rel_path) or "").strip().replace("\\", "/")
    path = os.path.normpath(os.path.join("data", *rel_path.split("/")))
    return path.replace("\\", "/")


def process_row(idx, row, model_kind, model_obj):
    image_path = resolve_image_path(row["image_path"])    
    ori_claim = row["claim"]
    updated = {}
    print(f"idx: {idx}")
    if pd.isna(row.get("veracity_0")):
        prompt = PMP_VERACITY.format(CLAIM=ori_claim)
        resp = prompt_commercial_model(model_obj, model_name, prompt, image_path, []) if model_kind=="commercial" else prompt_open_model(model_obj, prompt, image_path, [])
        updated["veracity_0"] = (resp or "").lstrip("Veracity:").strip()
    if pd.isna(row.get("veracity_evi")):
        evis = get_all_evis(image_path)
        prompt = PMP_VERACITY_EVI.format(CLAIM=ori_claim, EVIDENCE=evis)
        resp = prompt_commercial_model(model_obj, model_name, prompt, image_path, []) if model_kind=="commercial" else prompt_open_model(model_obj, prompt, image_path, [])
        updated["veracity_evi"] = (resp or "").lstrip("Veracity:").strip()
    return idx, updated


def main(args):
    train_set, dev_set, test_set = dataset_loader(args.file, args.task.removesuffix("_multilingual"))
    print([len(x) for x in [train_set, dev_set, test_set]])


    res_folder = os.path.join(f"res_{args.task}")
    os.makedirs(res_folder, exist_ok=True)
    result_file = os.path.join(res_folder, f"res_{args.task}_{args.model}_{args.lang}.xlsx")

    if not os.path.exists(result_file):
        total = []
        for d in test_set:
            image_path = d["image_path"]
            
            if d.get("use_true_caption"):
                claim = d["true_caption"] if (args.lang=="english" or check_contains(d["multilingual_claim"], ["not enough information"])) else d["multilingual_true_caption"]
                claim_verdict = "true"
            else:
                claim = d["claim"] if (args.lang=="english" or check_contains(d["multilingual_claim"], ["not enough information"])) else d["multilingual_claim"]
                claim_verdict = d["verdict"]
            total.append({"claim": claim, "claim_verdict": claim_verdict, "image_path": d["image_path"], "claim_language": d.get("claim_language")})
    
        pd.DataFrame(total).to_excel(result_file, index=False)

    df = pd.read_excel(result_file)
    df = init_cols(df, ["veracity_0","veracity_evi","claim_language"])    

    # Fill claim_language even if result_file already exists
    lang_map = {d["image_path"]: d.get("claim_language") for d in test_set}
    for idx, row in df.iterrows():
        val = lang_map.get(row["image_path"])
        if val is not None and (pd.isna(row.get("claim_language")) or row.get("claim_language") is None):
            df.at[idx, "claim_language"] = val

    global model_name
    model_name = args.model.lower()
    if model_name in ["gpt4o","gemini"]:
        model_kind = "commercial"; model_obj = get_commercial_model(model_name)
    elif model_name in ["internvl","llama","qwenvl"]:
        model_kind = "open"; model_obj = get_open_model(model_name)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if model_kind=="commercial":
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            fut = {ex.submit(process_row, idx, row, model_kind, model_obj): idx for idx, row in df.iterrows() if idx<=args.nums}
            for f in concurrent.futures.as_completed(fut):
                idx, upd = f.result()
                for k,v in upd.items(): df.at[idx, k] = v
    else:
        for idx, row in df.iterrows():
            if idx>args.nums: break
            idx, upd = process_row(idx, row, model_kind, model_obj)
            for k,v in upd.items(): df.at[idx, k] = v

    df.to_excel(result_file, index=False, engine='openpyxl')


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, default="data/M4FC.json")
    p.add_argument("--model", type=str, default="gemini")
    p.add_argument("--task", type=str, default="verdict_prediction_multilingual")
    p.add_argument("--nums", type=int, default=3000)
    p.add_argument("--lang", type=str, default="multilingual", choices=["english","multilingual"]) 
    args = p.parse_args()
    main(args)