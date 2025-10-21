"""
Sampling script for ASR experiments. Generates samples from models and saves results.
"""
import os

import csv
from tqdm import tqdm
import pandas as pd

from transformers import set_seed, pipeline
import torch

from parser import get_mbr_parser
from utils import load_model, load_dataset, load_kwargs
from utils import sample_dir


def sample(
    dataset,
    tokenizer,
    model,
    src_lines,
    torch_device,
    n_lines,
    start_iter,
    n_samples,
    bsz,
    temperature,
    eps,
    topk,
    topp,
    do_sample,
    diversity_penalty,
    prompt,
    stop_tokens,
    model_n,
    discard_prob=False,
    max_new_tokens=None,
):
    """
    Generate samples from an ASR model and save results to CSV files.
    Args:
        dataset (str): Dataset name.
        tokenizer: Tokenizer object.
        model: Model object.
        src_lines (list): Source audio samples.
        torch_device (str): Device for model inference.
        n_lines (int): Number of lines to process.
        start_iter (int): Starting index for iteration.
        n_samples (int): Number of samples to generate.
        bsz (int): Batch size.
        temperature (float): Sampling temperature.
        eps (float): Epsilon cutoff.
        topk (int): Top-k sampling parameter.
        topp (float): Top-p sampling parameter.
        do_sample (int): Sampling flag.
        diversity_penalty (float): Diversity penalty for beam search.
        prompt (str): Prompt for generation.
        stop_tokens (list): List of stop tokens.
        model_n (str): Model name.
        discard_prob (bool): Discard probability flag.
        max_new_tokens (int, optional): Maximum new tokens to generate.
    Returns:
        None
    """
    n_batches = n_samples // bsz

    if do_sample == 0:
        if n_batches > 1:
            print("n_batches must be 1 for beam search. Setting n_batches to 1.")
        n_batches = 1
        if diversity_penalty < 0.000001:
            print("Running beam search as diversity penalty is zero.")
    elif do_sample < 0:
        if n_batches > 1:
            print("n_batches must be 1 for beam search. Setting n_batches to 1.")
        n_batches = 1
        print("Running beam search as do_sample is negative.")
    else:
        # Running sampling algorithm.
        bsz = 1
        n_batches = n_samples

    os.makedirs(os.path.join(sample_dir, dataset, model_n), exist_ok=True)

    model_kwargs = load_kwargs(dataset)
    if max_new_tokens > 0:
        model_kwargs["max_new_tokens"] = max_new_tokens

    if "kotoba-whisper-bilingual-v1.0" in model_n:
        if "ja-en" in dataset:
            lang = "en"
            task = "translate"
        else:
            assert "en-ja" in dataset
            lang = "ja"
            task = "translate"
    elif (
        ("reazonspeech" in dataset)
        or ("jsut" in dataset)
        or ("commonvoice-ja" in dataset)
    ):
        lang = "ja"
        task = "transcribe"
    elif (
        ("librispeech" in dataset)
        or ("ami-asr" in dataset)
        or ("voxpopuli" in dataset)
    ):
        lang = "en"
        task = "transcribe"
    elif "commonvoice-" in dataset:
        lang = dataset.split("-")[1]
        task = "transcribe"
    else:
        assert False

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer.tokenizer,
        feature_extractor=tokenizer.feature_extractor,
        torch_dtype=model.dtype,
        model_kwargs=model_kwargs,
        batch_size=bsz,
        generate_kwargs={
            "language": lang,
            "task": task,
            "do_sample": do_sample > 0,
            "temperature": temperature,
            "epsilon_cutoff": eps,
            "top_k": topk,
            "top_p": topp,
            "num_beams": bsz,
        },
    )

    for sample_id in tqdm(range(start_iter, n_lines)):
        if sample_id > len(src_lines):
            break

        set_seed(42)

        rows = []

        sample = src_lines[sample_id]

        for i in range(n_batches):
            input_sample = {
                "raw": sample["array"],
                "sampling_rate": sample["sampling_rate"],
            }
            if not do_sample:
                result = pipe(
                    input_sample,
                    return_timestamps=False,
                    generate_kwargs={
                        "num_beams": bsz,
                    },
                )
            else:
                result = pipe(input_sample, return_timestamps=False)

            rows.append((result["text"], 0.0))

        if temperature != 1.0:
            filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}_tmp-{:.2f}".format(
                sample_id, eps, topk, topp, temperature
            )
        elif do_sample > 0:
            filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}".format(
                sample_id, eps, topk, topp
            )
        elif do_sample == 0:
            filename = "{:04}_beam-{:02d}_divpen-{:.2f}".format(
                sample_id, bsz, diversity_penalty
            )
        elif do_sample < 0:
            filename = "{:04}_beam-{:02d}".format(sample_id, bsz)
        else:
            assert False

        outfilepath = os.path.join(sample_dir, dataset, model_n, filename)

        df = pd.DataFrame(rows, columns=["text", "probability"])
        df.to_csv(
            outfilepath,
            sep=",",
            escapechar="\\",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            float_format="%.32f",
        )


if __name__ == "__main__":
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model
    prompt_path = args.prompt
    n_lines = args.n_lines
    start_iter = args.start_iter

    n_samples = args.n_samples
    bsz = args.bsz
    temperature = args.temperature
    eps = args.eps
    topk = args.topk
    topp = args.topp
    do_sample = args.do_sample
    diversity_penalty = args.diversity_penalty

    max_new_tokens = args.max_new_tokens

    quantize = args.quantize

    discard_prob = args.discard_prob

    src_lines = load_dataset(dataset)
    tokenizer, model, model_name, stop_tokens = load_model(
        dataset, torch_device, model_name, quantize
    )

    sample(
        dataset,
        tokenizer,
        model,
        src_lines,
        torch_device,
        n_lines,
        start_iter,
        n_samples,
        bsz,
        temperature,
        eps,
        topk,
        topp,
        do_sample,
        diversity_penalty,
        "None",
        stop_tokens,
        model_n=os.path.basename(model_name),
        discard_prob=discard_prob,
        max_new_tokens=max_new_tokens,
    )
