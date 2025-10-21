"""
Main engine for MBR ASR experiments. Loads data, computes scores, and runs the experiment pipeline.
"""
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

from utility_func import load_evaluate, evaluate_diversity, load_similarity
from utils import (
    load_dataset,
    load_matrix,
    load_samples_from_file,
    result_dir,
    matrix_dir,
)
from parser import get_mbr_parser
from policy.mbr import compute_mbr


def compute_score(df, d_best, trg, compute_evaluate, src=None):
    d_hyp = df.iloc[d_best]["text"]
    d_score = compute_evaluate(d_hyp, trg, src)
    return d_score


if __name__ == "__main__":
    """
    Compute the score for the best hypothesis in a DataFrame using the evaluation function.
    Args:
        df (pd.DataFrame): DataFrame containing hypotheses.
        d_best (int): Index of the best hypothesis.
        trg (str): Target/reference string.
        compute_evaluate (callable): Evaluation function.
        src (optional): Source input.
    Returns:
        float: Computed score.
    """
    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model

    sample_dir = args.sample_dir
    matrix_dir = args.matrix_dir

    n_lines = args.n_lines
    start_iter = args.start_iter
    n_samples = args.n_samples

    temperature = args.temperature
    epsilon = args.eps
    topk = args.topk
    topp = args.topp

    sim = args.sim
    eval_func = args.eval

    # Algorithm config
    algorithm = args.algorithm
    recompute_matrix = args.recompute_matrix
    approx_iters = args.approx_iters
    r_0 = args.r_0
    r_increase = args.r_increase
    pruning_alpha = args.pruning_alpha

    diverse_k = args.diverse_k
    diversity_penalty = args.diversity_penalty
    pairwise_eval = args.pairwise_eval

    do_sample = args.do_sample

    if algorithm == "dbs":
        assert do_sample == 0
    elif algorithm == "beam":
        assert do_sample < 0
    else:
        assert do_sample

    compute_evaluate, evaluator = load_evaluate(eval_func, None, None)

    if algorithm in ["dbs", "diverse", "diversesample", "oversampling"]:
        compute_pairwise, _ = load_evaluate(pairwise_eval, None, None)
    else:
        compute_pairwise = None

    src_lines = load_dataset(dataset)  # src is used only by comet and clip.
    trg_lines = load_dataset(dataset, ref=True)

    model_n = os.path.basename(model_name)

    os.makedirs(os.path.join(matrix_dir, dataset, model_n), exist_ok=True)

    files = sorted(os.listdir(sample_dir))
    if do_sample > 0:
        print(
            "filter files with epsilon={}, topk={}, topp={}, temperature={}".format(
                epsilon, topk, topp, temperature
            )
        )
    else:
        print(
            "filter files with diverse_k={}, diverse_penalty={}".format(
                diverse_k, diversity_penalty
            )
        )
    filtered_files = load_samples_from_file(
        files, epsilon, topk, topp, do_sample, diverse_k, diversity_penalty, temperature
    )

    assert len(filtered_files) > 0

    print("first 10 files=", filtered_files[:10])

    rows = []

    for sample_id in tqdm(range(start_iter, n_lines)):
        if sample_id > len(src_lines):
            break
        filename = filtered_files[sample_id]
        if not (
            ("{:04}".format(sample_id) in filename)
            or ("{:05}".format(sample_id) in filename)
        ):
            print(
                "Error: sample_id mismatch: sample_id=",
                sample_id,
                "filename=",
                filename,
            )
        assert ("{:04}".format(sample_id) in filename) or (
            "{:05}".format(sample_id) in filename
        )

        src_input = src_lines[sample_id]
        trg = trg_lines[sample_id]

        try:
            df = pd.read_csv(os.path.join(sample_dir, filename))
        except Exception as e:
            print(
                os.path.join(sample_dir, filename),
                "is not readable with default engine.",
                "Error:",
                e
            )
            continue

        if algorithm not in ["dbs", "beam"]:
            assert len(df) >= n_samples
            df = df[:n_samples]
        df.fillna("", inplace=True)
        df["text"] = df["text"].astype(str)
        hyp = df.iloc[:]["text"]

        if algorithm not in ["dbs", "beam", "mean"]:
            if not recompute_matrix:
                matrix = load_matrix(
                    os.path.join(matrix_dir, dataset, model_n), filename, sim, n_samples
                )
            else:
                matrix = None
            if matrix is None:
                matrix_filename = filename + "_" + sim + "_" + str(n_samples)
                matrix_path = os.path.join(
                    matrix_dir, dataset, model_n, matrix_filename
                )

                compute_similarity = load_similarity(sim)
                matrix = compute_similarity.compute_score_matrix(hyp, src_input)

                np.savetxt(matrix_path, matrix)

        if algorithm == "base":
            # MBR: Monte Carlo Estimate
            ed_best = compute_mbr(matrix=matrix)
            ed_score = compute_score(df, ed_best, trg, compute_evaluate, src=src_input)
            row = [sample_id, ed_score, ed_best]

        elif algorithm in ["None", "none", "exact"]:
            # MBR: Monte Carlo Estimate
            ed_best = compute_mbr(matrix=matrix)
            ed_score = compute_score(df, ed_best, trg, compute_evaluate, src=src_input)

            row = [sample_id, ed_score, ed_best]
        elif algorithm == "diversesample":
            dbs_hyps = df["text"].to_list()[:diverse_k]
            dbs_scores = [
                compute_score(df, i, trg, compute_evaluate, src=src_input)
                for i in range(diverse_k)
            ]
            dbs_stats = evaluate_diversity(
                dbs_hyps, dbs_scores, src_input, compute_pairwise
            )
            row = [sample_id, dbs_scores, []] + dbs_stats
        elif algorithm == "dbs":
            dbs_hyps = df["text"].to_list()
            dbs_scores = [
                compute_score(df, i, trg, compute_evaluate, src=src_input)
                for i in range(diverse_k)
            ]
            dbs_stats = evaluate_diversity(
                dbs_hyps, dbs_scores, src_input, compute_pairwise
            )
            row = [sample_id, dbs_scores, []] + dbs_stats
        elif algorithm == "beam":
            bs_hyps = df["text"].to_list()
            bs_score = compute_score(df, 0, trg, compute_evaluate, src=src_input)
            # bs_stats = evaluate_diversity(bs_hyps, bs_scores, src_input, compute_pairwise)
            row = [sample_id, bs_score, []]
        else:
            assert False
        rows.append(row)

    if (algorithm == "base") or (algorithm == "exact") or (algorithm == "None"):
        columns = ["sample_id", "mbr_score", "selected_id"]
        postfix = ""
    elif algorithm == "diversesample":
        columns = ["sample_id"]
        methods = ["diversesample-{}".format(diversity_penalty)]
        metrics = [
            "_score",
            "_best",
            "_mean_score",
            "_min_score",
            "_max_score",
            "_self_score",
            "_dn_1",
            "_dn_2",
            "_dn_3",
        ]
        for method in methods:
            cl = [method + metric for metric in metrics]
            columns += cl
        postfix = "_diversesample_{:02d}".format(diverse_k)

        if pairwise_eval != "sacrebleu":
            postfix += "_{}".format(pairwise_eval)
    elif algorithm == "dbs":
        columns = ["sample_id"]
        methods = ["dbs-{}".format(diversity_penalty)]
        metrics = [
            "_score",
            "_best",
            "_mean_score",
            "_min_score",
            "_max_score",
            "_self_score",
            "_dn_1",
            "_dn_2",
            "_dn_3",
        ]
        for method in methods:
            cl = [method + metric for metric in metrics]
            columns += cl
        postfix = "_dbs_{:02d}_{:.2f}".format(diverse_k, diversity_penalty)
        if pairwise_eval != "sacrebleu":
            postfix += "_{}".format(pairwise_eval)
    elif algorithm == "beam":
        columns = ["sample_id", "beam_score", "beam_best"]
        postfix = "_beam_{:02d}".format(diverse_k)
    else:
        assert False

    df = pd.DataFrame(rows, columns=columns)

    if algorithm == "dbs":
        filename = "{}_{}_{}{}.csv".format(dataset, model_n, eval_func, postfix)
    elif algorithm == "beam":
        filename = "{}_{}_{}{}.csv".format(dataset, model_n, eval_func, postfix)
    elif temperature != 1.0:
        filename = "{}_{}_{:03}_{:.2f}_{:02d}_{:.2f}_{:.2f}_{}_{}{}.csv".format(
            dataset,
            model_n,
            n_samples,
            epsilon,
            topk,
            topp,
            temperature,
            sim,
            eval_func,
            postfix,
        )
    else:
        filename = "{}_{}_{:03}_{:.2f}_{:02d}_{:.2f}_{}_{}{}.csv".format(
            dataset, model_n, n_samples, epsilon, topk, topp, sim, eval_func, postfix
        )

    df_path = os.path.join(result_dir, filename)
    df.to_csv(df_path, index=False)
