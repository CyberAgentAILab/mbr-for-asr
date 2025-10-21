"""
Utility functions and constants for ASR experiments, including model loading,
dataset management, and configuration.
"""
import os
import random
from collections.abc import Iterable
from glob import glob

import numpy as np

import torch
import datasets
from datasets import Audio
import torchaudio

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

dataset_dir = "./dataset"
sample_dir = "./samples"
score_dir = "./score"  # not used
output_dir = "./output"  # not used
evaluate_dir = "./evaluate"
prompt_dir = "./prompts"
result_dir = "./results"
matrix_dir = "./matrix"
embed_dir = "./embed"
instruct_dir = "./instruct"

reward_dir = "./reward"

HF_READ_TOKEN = os.getenv("HF_READ_TOKEN", "auto")

audio_datasets = [
    "librispeech-asr-test",
    "commonvoice-ja",
    "jsut",
    "reazonspeech",
    "covost2-ja-en",
    "fleurs-ja-en",
    "covost2-en-ja",
    "fleurs-en-ja",
]


def load_model(dataset, torch_device, model_name, quantize=-1):
    """
    Load a model and tokenizer for the specified dataset and model name.

    Args:
        dataset (str): Name of the dataset.
        torch_device (str): Device to load the model on (e.g., 'cpu', 'cuda').
        model_name (str): Name of the model to load.
        quantize (int): Quantization level (-1 for none, 4 or 8 for quantized).

    Returns:
        tuple: (tokenizer, model, model_name, stop_tokens)
    """
    from transformers import FSMTForConditionalGeneration, FSMTTokenizer
    from transformers import BartForConditionalGeneration, BartTokenizer
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    from transformers import AutoModel, AutoProcessor, Blip2ForConditionalGeneration
    from transformers import (
        PaliGemmaForConditionalGeneration,
        AutoModelForSpeechSeq2Seq,
    )

    q4 = quantize == 4
    q8 = quantize == 8

    stop_tokens = []

    model_n = os.path.basename(model_name)
    base_model_name = model_name

    if ("whisper" in model_n) or ("distil-large-v3.5" in model_n):
        if "lite-whisper-large-v3-turbo" in model_n:
            tokenizer = AutoProcessor.from_pretrained(
                "openai/whisper-large-v3", trust_remote_code=True, token=HF_READ_TOKEN
            )
        else:
            tokenizer = AutoProcessor.from_pretrained(
                base_model_name, trust_remote_code=True, token=HF_READ_TOKEN
            )

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
            token=HF_READ_TOKEN,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if "lite-whisper" in base_model_name:
        model = AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            load_in_4bit=q4,
            load_in_8bit=q8,
            device_map="auto",
            torch_dtype="auto",
            token=HF_READ_TOKEN,
        )
    elif ("whisper" in base_model_name) or ("distil-large-v3.5" in base_model_name):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            load_in_4bit=q4,
            load_in_8bit=q8,
            device_map="auto",
            torch_dtype="auto",
            token=HF_READ_TOKEN,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            load_in_4bit=q4,
            load_in_8bit=q8,
            device_map="auto",
            torch_dtype="auto",
            token=HF_READ_TOKEN,
        )

    model.eval()

    return tokenizer, model, model_name, stop_tokens


def load_dataset(dataset, ref=False, raw_text=False):
    """
    Load dataset samples for ASR experiments.
    Args:
        dataset (str): Dataset name.
        ref (bool): If True, return reference texts; else return audio.
        raw_text (bool): If True, return raw text.
    Returns:
        list: List of audio samples or reference texts.
    """

    if "librispeech-asr-test" in dataset:
        dss = datasets.load_dataset(
            "ddyuudd/librispeech", split="test", streaming=True
        ).take(1000)
        if not ref:
            lines = [d["audio"] for d in dss]
        else:
            lines = [d["text"] for d in dss]
    elif "ami-asr" in dataset:
        dss = datasets.load_dataset(
            "edinburghcstr/ami", "ihm", split="test", streaming=True
        ).take(1000)
        if not ref:
            lines = [d["audio"] for d in dss]
        else:
            lines = [d["text"] for d in dss]
    elif "voxpopuli-asr" in dataset:
        dss = datasets.load_dataset(
            "ddyuudd/voxpopuli", split="train", streaming=True
        ).take(1000)
        if not ref:
            lines = [d["audio"] for d in dss]
        else:
            lines = [d["text"] for d in dss]
    elif "commonvoice-ja" in dataset:
        dss = datasets.load_dataset(
            "japanese-asr/ja_asr.common_voice_8_0", split="test", streaming=True
        ).take(1000)
        dss = dss.cast_column("audio", Audio(sampling_rate=16000))
        if not ref:
            lines = [d["audio"] for d in dss]
        else:
            lines = [d["transcription"] for d in dss]
    elif "commonvoice-" in dataset:
        if len(dataset.split("-")) > 2:
            lang = dataset.split("-")[1] + "-" + dataset.split("-")[2]
        else:
            lang = dataset.split("-")[1]
        dss = datasets.load_dataset(
            "mozilla-foundation/common_voice_8_0",
            lang,
            trust_remote_code=True,
            streaming=True,
        )["test"].take(1000)
        dss = dss.cast_column("audio", Audio(sampling_rate=16000))
        if not ref:
            lines = [d["audio"] for d in dss]
        else:
            lines = [d["sentence"] for d in dss]
    elif "jsut" in dataset:
        dss = datasets.load_dataset(
            "japanese-asr/ja_asr.jsut_basic5000", split="test", streaming=True
        ).take(1000)
        dss = dss.cast_column("audio", Audio(sampling_rate=16000))
        if not ref:
            lines = [d["audio"] for d in dss]
        else:
            lines = [d["transcription"] for d in dss]
    elif "reazonspeech" in dataset:
        dss = datasets.load_dataset(
            "japanese-asr/ja_asr.reazonspeech_test", split="test", streaming=True
        ).take(1000)
        if not ref:
            lines = [d["audio"] for d in dss]
        else:
            lines = [d["transcription"] for d in dss]
    elif "covost2-ja-en" in dataset:
        dss = datasets.load_dataset(
            "japanese-asr/ja2en.s2t_translation",
            "covost2",
            split="test",
            streaming=True,
        ).take(1000)
        if not ref:
            lines = [d["audio"] for d in dss]
        else:
            lines = [d["translation"] for d in dss]
    elif "fleurs-ja-en" in dataset:
        dss = datasets.load_dataset(
            "japanese-asr/ja2en.s2t_translation", "fleurs", split="test", streaming=True
        ).take(1000)
        if not ref:
            lines = [d["audio"] for d in dss]
        else:
            lines = [d["translation"] for d in dss]
    elif "covost2-en-ja" in dataset:
        dss = datasets.load_dataset(
            "japanese-asr/en2ja.s2t_translation",
            "covost2",
            split="test",
            streaming=True,
        ).take(1000)
        if not ref:
            lines = [d["audio"] for d in dss]
        else:
            lines = [d["translation"] for d in dss]
    elif "fleurs-en-ja" in dataset:
        dss = datasets.load_dataset(
            "japanese-asr/en2ja.s2t_translation", "fleurs", split="test", streaming=True
        ).take(1000)
        if not ref:
            lines = [d["audio"] for d in dss]
        else:
            lines = [d["translation"] for d in dss]
    else:
        assert False, "Dataset {} not implemented".format(dataset)

    # Add noise if the dataset name ends with e.g. -5db, -10db, -15db, -20db
    if (not ref) and any([audio_d in dataset for audio_d in audio_datasets]):
        if dataset[-2:] == "db":
            noise_db = int(dataset.split("-")[-1][:-2].replace("m", "-"))
            print("dataset=", dataset)
            print("Noise dB:", noise_db)
            lines = synthesize_audio_noise(lines, noise_db)
        else:
            print("dataset=", dataset)
            print("clean (no noise)")

    assert isinstance(lines, Iterable)
    return lines


def load_kwargs(dataset):
    """
    Return keyword arguments for model loading for a given dataset.
    Args:
        dataset (str): Dataset name.
    Returns:
        dict: Keyword arguments for model loading.
    """
    return dict()


def load_matrix(target_matrix_dir, filename, sim, n_samples):
    """
    Load similarity or score matrix from directory.
    Args:
        target_matrix_dir (str): Directory containing matrix files.
        filename (str): Base filename.
        sim (str): Similarity metric name.
        n_samples (int): Number of samples to truncate matrix to.
    Returns:
        np.ndarray or None: Loaded matrix or None if not found.
    """

    matrix_base = os.path.join(target_matrix_dir, filename + "_" + sim + "_")
    matrix_paths = glob(matrix_base + "*")

    cached_nsamples = [int(f[len(matrix_base) :]) for f in matrix_paths]
    larger_cahces = [c for c in cached_nsamples if c >= n_samples]

    if len(larger_cahces) == 0:
        return None

    min_nsamples = min(larger_cahces)

    matrix = np.loadtxt(matrix_base + str(min_nsamples))
    matrix = matrix[:n_samples, :n_samples]

    return matrix


def load_samples_from_file(
    files, epsilon, topk, topp, do_sample, diverse_k, divpen, temperature=1.0
):
    """
    Filter sample files based on sampling parameters.
    Args:
        files (list): List of filenames.
        epsilon (float): Epsilon value for filtering.
        topk (int): Top-k value for filtering.
        topp (float): Top-p value for filtering.
        do_sample (int): Sampling flag.
        diverse_k (int): Beam diversity parameter.
        divpen (float): Diversity penalty.
        temperature (float): Sampling temperature.
    Returns:
        list: Filtered filenames.
    """
    # To keep backward compatibility to the old format, it needs two steps.
    # First it loads in current format and it no files found, it loads in old format.
    filtered_files = []

    if do_sample > 0:
        for filename in files:
            isnt_eps = not "eps-{:.2f}".format(epsilon) in filename

            # If topk is set to negative (e.g. -1), then it means that "topk" should not be in the filename.
            if topk < 0:
                isnt_topk = "topk" in filename
            else:
                isnt_topk = not "topk-{:02d}".format(topk) in filename

            if topp < 0:
                isnt_topp = "topp" in filename
            else:
                isnt_topp = not "topp-{:.2f}".format(topp) in filename

            if not (isnt_eps or isnt_topk or isnt_topp):
                filtered_files.append(filename)
        filtered_files.sort(key=lambda x: int(x.split("_eps")[0]))
    elif do_sample == 0:
        for filename in files:
            k_matches = "beam-{:02d}".format(diverse_k) in filename
            dp_matches = "divpen-{:.2f}".format(divpen) in filename

            if k_matches and dp_matches:
                filtered_files.append(filename)
    else:
        for filename in files:
            k_matches = "beam-{:02d}".format(diverse_k) in filename
            dp_matches = not "divpen" in filename

            if k_matches and dp_matches:
                filtered_files.append(filename)

    return filtered_files


def list_to_text(words):
    """
    Convert a list of words to a single space-separated string.
    Args:
        words (list): List of words.
    Returns:
        str: Space-separated string.
    """
    text = words[0]
    for w in words[1:]:
        text = text + " " + w
    return text


#############################
# Speech Processing Utilities
def add_noise(clean_waveform, noise_waveform, snr_db):
    """
    Add noise to a clean waveform at a specified SNR.
    Args:
        clean_waveform (Tensor): Clean audio waveform.
        noise_waveform (Tensor): Noise audio waveform.
        snr_db (float): Desired signal-to-noise ratio in dB.
    Returns:
        Tensor: Noisy waveform.
    """
    clean_power = clean_waveform.norm(p=2)
    noise_power = noise_waveform.norm(p=2)
    factor = (clean_power / noise_power) * 10 ** (-snr_db / 20)
    return clean_waveform + factor * noise_waveform


def get_random_noise(num_samples, noise_dataset, sample_rate=16000, index=None):
    """
    Get a random noise waveform from a noise dataset, matching the desired length.
    Args:
        num_samples (int): Number of samples for output noise.
        noise_dataset (Dataset): Dataset containing noise samples.
        sample_rate (int): Desired sample rate.
        index (int, optional): Specific index to use; if None, random.
    Returns:
        Tensor: Noise waveform of desired length.
    """
    if index is None:
        index = random.randint(0, len(noise_dataset))  # Randomly select noise index
    noise_waveform = noise_dataset[index]["audio"]["array"]
    noise_waveform = torch.tensor(noise_waveform).unsqueeze(0)

    # Pad or crop noise to match desired duration
    if noise_waveform.shape[1] >= num_samples:
        start = random.randint(0, noise_waveform.shape[1] - num_samples)
        noise_waveform = noise_waveform[:, start : start + num_samples]
    else:
        num_repeats = (
            num_samples + noise_waveform.shape[1] - 1
        ) // noise_waveform.shape[1]
        noise_waveform = noise_waveform.repeat(1, num_repeats)[:, :num_samples]

    return noise_waveform


def synthesize_audio_noise(dataset, snr_db):
    """
    Synthesize noisy audio samples by adding noise to each sample in a dataset.
    Args:
        dataset (list): List of audio samples (dicts with 'array' and 'sampling_rate').
        snr_db (float): Desired signal-to-noise ratio in dB.
    Returns:
        list: List of noisy audio samples.
    """
    TARGET_SAMPLE_RATE = 16000  # Target sample rate for audio processing

    noise_dataset = datasets.load_dataset("ddyuudd/musan_free_sound", split="train")
    noisy_dataset = []
    for i, example in enumerate(dataset):
        waveform = example["array"]
        waveform = torch.tensor(waveform).unsqueeze(0)
        sr = example["sampling_rate"]

        # Resample to 16kHz if needed
        if sr != TARGET_SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(waveform)

        # duration_sec = waveform.shape[1] / TARGET_SAMPLE_RATE
        noise = get_random_noise(
            waveform.shape[1],
            noise_dataset,
            TARGET_SAMPLE_RATE,
            index=i % len(noise_dataset),
        )

        noisy = add_noise(waveform, noise, snr_db)
        noisy_dataset.append(
            {"array": noisy.squeeze(0).numpy(), "sampling_rate": TARGET_SAMPLE_RATE}
        )

    return noisy_dataset
