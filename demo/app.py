import os
from math import floor
from copy import deepcopy
from typing import Optional, List, Dict, Any
import numpy as np
import spaces
import torch
import gradio as gr
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import torch.nn.functional as F
from sacrebleu.metrics import BLEU

# Configuration
MODEL_NAME = "kotoba-tech/kotoba-whisper-v2.0"
BATCH_SIZE = 16
CHUNK_LENGTH_S = 15
MAX_AUDIO_LENGTH = 30  # Maximum audio length in seconds

# Device setting
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    device = "cuda"
    model_kwargs = {"attn_implementation": "sdpa"}
else:
    torch_dtype = torch.float32
    device = "cpu"
    model_kwargs = {}

# Define the pipeline
pipe = pipeline(
    model=MODEL_NAME,
    chunk_length_s=CHUNK_LENGTH_S,
    batch_size=BATCH_SIZE,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs=model_kwargs,
    trust_remote_code=True,
)


def format_time(start: Optional[float], end: Optional[float]):
    """Format timestamp for display"""

    def _format_time(seconds: Optional[float]):
        if seconds is None:
            return "complete    "
        minutes = floor(seconds / 60)
        hours = floor(seconds / 3600)
        seconds = seconds - hours * 3600 - minutes * 60
        m_seconds = floor(round(seconds - floor(seconds), 3) * 10**3)
        seconds = floor(seconds)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{m_seconds:03}"

    return f"[{_format_time(start)}-> {_format_time(end)}]:"


def check_audio_length(inputs):
    """Check if audio length is within the 30-second limit"""
    if inputs is None:
        return None

    with open(inputs, "rb") as f:
        audio_data = f.read()

    audio_array = ffmpeg_read(audio_data, pipe.feature_extractor.sampling_rate)
    duration = len(audio_array) / pipe.feature_extractor.sampling_rate

    if duration > MAX_AUDIO_LENGTH:
        # Truncate to 30 seconds
        max_samples = int(MAX_AUDIO_LENGTH * pipe.feature_extractor.sampling_rate)
        audio_array = audio_array[:max_samples]

    return audio_array, duration


@spaces.GPU
def beam_search_transcription(inputs, num_beams: int = 5):
    """Perform beam search transcription"""

    print("beam_search: inputs=", inputs)
    generate_kwargs = {
        "language": "en",
        "task": "transcribe",
        "num_beams": num_beams,
        "do_sample": False,
    }

    # if prompt:
    #     generate_kwargs['prompt_ids'] = pipe.tokenizer.get_prompt_ids(prompt, return_tensors='pt').to(device)

    prediction = pipe(inputs, return_timestamps=True, generate_kwargs=generate_kwargs)
    text = "".join([c["text"] for c in prediction["chunks"]])
    text_timestamped = "\n".join(
        [f"{format_time(*c['timestamp'])} {c['text']}" for c in prediction["chunks"]]
    )

    return text, text_timestamped


@spaces.GPU
def mbr_decoding_transcription(inputs, num_candidates: int = 10):
    """Perform MBR (Minimum Bayes Risk) decoding transcription"""

    print("mbr_decoding: inputs=", inputs)

    # Generate multiple candidates using pipeline calls with sampling
    candidates = []

    # Updated generate_kwargs for pipeline usage
    pipeline_kwargs = {
        "language": "en",
        "task": "transcribe",
        "do_sample": True,
        "temperature": 1.0,
        "epsilon_cutoff": 0.01,
    }

    # Generate candidates by calling pipeline multiple times
    for i in range(num_candidates):
        inputs_ = deepcopy(inputs)
        try:
            prediction = pipe(
                inputs_, return_timestamps=True, generate_kwargs=pipeline_kwargs
            )
            candidate_text = "".join([c["text"] for c in prediction["chunks"]])
            candidates.append(candidate_text)
        except Exception as e:
            print(f"Error generating candidate {i}: {e}")
            # If generation fails, add empty string to maintain list length
            candidates.append("")

    # Perform MBR decoding: select candidate with minimum expected risk
    # Risk is computed as negative similarity to other candidates
    mbr_scores = []

    for i, candidate_i in enumerate(candidates):
        utility = 0.0
        for j, candidate_j in enumerate(candidates):
            if i != j:
                # Simple similarity metric: character-level overlap
                similarity = compute_similarity(candidate_i, candidate_j)
                utility += similarity
        mbr_scores.append(utility / (len(candidates) - 1))

    # Select candidate with minimum risk (maximum score)
    best_idx = max(range(len(mbr_scores)), key=lambda i: mbr_scores[i])
    best_candidate = candidates[best_idx]

    return (
        best_candidate,
        f"MBR Selected: {best_candidate}\n\nAll Candidates:\n"
        + "\n".join([f"{i+1}. {cand}" for i, cand in enumerate(candidates)]),
    )


def compute_similarity(candidate: str, reference: str) -> float:
    """Compute character-level similarity between two texts"""
    assert isinstance(candidate, str)
    assert isinstance(reference, str)
    if not candidate or not reference:
        return 0.0

    bleu = BLEU()
    # bleu = BLEU(tokenize='ja-mecab')
    score = bleu.corpus_score([candidate], [[reference]])

    return score.score


def transcribe_comparison(inputs: str, num_beams: int, num_candidates: int):
    """Compare beam search and MBR decoding results"""
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request."
        )

    # Check audio length and process
    result = check_audio_length(inputs)
    if result is None:
        raise gr.Error("Failed to process audio file.")

    audio_array, duration = result

    # Add padding
    array_pad = np.zeros(int(pipe.feature_extractor.sampling_rate * 0.5))
    audio_array = np.concatenate([array_pad, audio_array, array_pad])
    audio_inputs = {
        "array": audio_array,
        "sampling_rate": pipe.feature_extractor.sampling_rate,
    }

    assert isinstance(audio_inputs, dict)
    assert "array" in audio_inputs and "sampling_rate" in audio_inputs
    # Perform beam search transcription
    beam_text, beam_timestamped = beam_search_transcription(audio_inputs, num_beams)

    # Rerun this code as pipe seems to have side effect on inputs.
    array_pad = np.zeros(int(pipe.feature_extractor.sampling_rate * 0.5))
    audio_array = np.concatenate([array_pad, audio_array, array_pad])
    audio_inputs = {
        "array": audio_array,
        "sampling_rate": pipe.feature_extractor.sampling_rate,
    }
    assert isinstance(audio_inputs, dict)
    assert "array" in audio_inputs and "sampling_rate" in audio_inputs
    # Perform MBR decoding transcription
    mbr_text, mbr_details = mbr_decoding_transcription(audio_inputs, num_candidates)

    # Create comparison output
    duration_info = f"Audio Duration: {duration:.2f} seconds (max: {MAX_AUDIO_LENGTH}s)"

    comparison = f"""
## Transcription Comparison

**Audio Duration:** {duration:.2f} seconds

### Beam Search Result (num_beams={num_beams})
{beam_text}

### MBR Decoding Result (candidates={num_candidates})
{mbr_text}

### Detailed Beam Search (with timestamps)
{beam_timestamped}

### MBR Decoding Details
{mbr_details}
"""

    return comparison, beam_text, mbr_text


# Create Gradio interface
with gr.Blocks(title="MBR ASR Demo") as demo:
    gr.Markdown("""
    # MBR ASR Transcription Demo
    
    This demo compares **Beam Search** and **Minimum Bayes Risk (MBR) Decoding** for English ASR transcription.
    
    - **Beam Search**: Traditional decoding that finds the most likely sequence
    - **MBR Decoding**: Selects the candidate with minimum expected risk among multiple generated sequences
    
    **Note**: Audio is automatically limited to 30 seconds for this demo.
    """)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio Input (max 30 seconds)",
            )

            with gr.Row():
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Beams for Beam Search",
                )
                num_candidates = gr.Slider(
                    minimum=1,
                    maximum=64,
                    value=8,
                    step=1,
                    label="Samples for MBR Decoding",
                )

            transcribe_btn = gr.Button(
                "Transcribe by Beam search and MBR Decoding", variant="primary"
            )

        with gr.Column():
            comparison_output = gr.Markdown(label="Comparison Results")

    with gr.Row():
        beam_output = gr.Textbox(label="Beam Search Result", lines=3)
        mbr_output = gr.Textbox(label="MBR Decoding Result", lines=3)

    transcribe_btn.click(
        fn=transcribe_comparison,
        inputs=[audio_input, num_beams, num_candidates],
        outputs=[comparison_output, beam_output, mbr_output],
    )

    gr.Markdown("""
    ## About the Methods
    
    **Beam Search**: Maintains the top-k most likely partial sequences at each step, selecting the single best complete sequence.
    
    **MBR Decoding**: Generates multiple candidate sequences and selects the one that minimizes expected risk (maximizes similarity to other candidates), potentially producing more robust results.
    
    **Model**: [kotoba-tech/kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)
    """)

if __name__ == "__main__":
    demo.queue(api_open=False, default_concurrency_limit=40).launch(
        show_api=False, show_error=True
    )
