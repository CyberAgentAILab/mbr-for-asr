import argparse


def get_mbr_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name e.g. wmt19.de-en, xsum, nocaps")
    parser.add_argument(
        "--n_lines",
        type=int,
        default=4,
        help="number of source inputs to evaluate. default is 4 so that it can be used for debugging",
    )
    parser.add_argument("--start_iter", type=int, default=0, help="starting id")

    parser.add_argument(
        "--model",
        default="None",
        help="default is None which is to select predefined model for each dataset",
    )
    parser.add_argument(
        "--prompt", default="None", help="only applicable for Language models"
    )
    parser.add_argument(
        "--quantize",
        default=-1,
        type=int,
        help="quantize the input to the model. -1 is no quantization",
    )

    # Sampling algorithm
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="For sample.py, number of samples to generate for sampling algorithm. "
        + "For mbr_engine.py, this is the number of samples to generate for each source input",
    )
    parser.add_argument(
        "--bsz", type=int, default=4, help="batch size for sampling algorithm"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature for sampling algorithm",
    )
    parser.add_argument(
        "--eps", type=float, default=0.02, help="epsilon for sampling algorithm"
    )
    parser.add_argument(
        "--topk", type=int, default=0, help="topk for sampling algorithm"
    )
    parser.add_argument(
        "--topp", type=float, default=1.0, help="topp for sampling algorithm"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=-1,
        help="max new tokens for sampling algorithm",
    )

    parser.add_argument(
        "--discard_prob",
        action="store_true",
        help="whether to discard prob for sampling algorithm",
    )

    # MBR Algorithm
    parser.add_argument("--sample_dir", help="directory to save samples")
    parser.add_argument(
        "--matrix_dir", default="./matrix", help="directory to save similarity matrix"
    )

    parser.add_argument("--algorithm", default="None", help="mbr algorithm")
    parser.add_argument(
        "--recompute_matrix",
        action="store_true",
        help="whether to recompute similarity matrix",
    )

    # Utility function
    parser.add_argument(
        "--sim",
        default="bertscore",
        help="similarity function (utility function) for MBR",
    )
    # Evaluation function
    parser.add_argument(
        "--eval", default="bleu", help="quality metric for evaluating the output"
    )

    # Approximate MBR
    parser.add_argument(
        "--approx_iters",
        type=int,
        default=5,
        help="number of runs for approximate MBR (AMBR is non-deterministic algorithm)",
    )
    parser.add_argument(
        "--approx_budgets", type=int, default=-1, help="budgets for approximate MBR"
    )

    # Pruning MBR
    parser.add_argument(
        "--r_0",
        type=int,
        default=4,
        help="size of the reference set at the first iteration for pruning MBR",
    )
    parser.add_argument(
        "--r_increase",
        type=int,
        default=2,
        help="increase rate of the reference set for pruning MBR",
    )
    parser.add_argument(
        "--pruning_alpha",
        type=float,
        default=0.9,
        help="Hypothesis with win rate less than 1 - pruning_alpha will be pruned",
    )

    ### Diverse MBR
    parser.add_argument(
        "--diverse_k",
        type=int,
        default=4,
        help="number of output sequences for diverse MBR",
    )
    # parser.add_argument('--diverse_pen', type=float, default=0.5, help='diversity penalty for diverse MBR')
    parser.add_argument(
        "--diversity_penalty",
        type=float,
        default=1.0,
        help="diversity penalty for sampling algorithm",
    )
    parser.add_argument(
        "--pairwise_eval",
        type=str,
        default="sacrebleu",
        help="diversity evaluation metric",
    )

    # This is for beam search and diverse beam search
    parser.add_argument(
        "--do_sample", type=int, default=1, help="beam size for sampling algorithm"
    )

    #################
    # Reward modeling
    parser.add_argument(
        "--reward_model",
        default="OpenAssistant/reward-model-deberta-v3-large-v2",
        help="reward model",
    )
    parser.add_argument(
        "--recompute_reward", action="store_true", help="whether to recompute reward"
    )
    parser.add_argument(
        "--compared_dir",
        type=str,
        default=None,
        help="sample_dir for the model to be compared for pairwise comparison reward model",
    )
    parser.add_argument(
        "--c_nsamples",
        type=int,
        default=1,
        help="number of samples for the model to be compared for pairwise comparison reward model",
    )

    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="compute pairwise comparison of reward using pairwise reward model",
    )
    parser.add_argument(
        "--get_raw",
        action="store_true",
        help="get raw text from a LLM as a judge based reward model",
    )

    #################
    # Instruction Pruning
    parser.add_argument(
        "--pruning_method",
        type=str,
        default="incremental-pruning",
        help="pruning method for instruction subset selection",
    )
    parser.add_argument(
        "--pruning_threshold",
        type=float,
        default=0.7,
        help="pruning threshold for instruction subset selection",
    )
    parser.add_argument(
        "--n_output_instructions",
        type=int,
        default=100,
        help="number of instructions to select for instruction subset selection",
    )

    return parser
