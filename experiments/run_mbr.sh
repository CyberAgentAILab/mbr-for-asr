# Default parameters are set to run a debug experiment.

DOMAIN=librispeech-asr-test
MODEL=openai/whisper-tiny # openai/whisper-large-v3
NLINES=2
NSAMPLES=4
TEMPERATURE=1.0
EPS=0.01
TOPK=0
TOPP=1.0
SIM=sacrebleu
EVAL=wer

ALGORITHM=None
DEBUG=0
RECOMPUTE=""

RZERO=4
PALPHA=0.9
DOSAMPLE=1
DIVERSITY=1.0
DIVERSEK=4
PAIRWISE=sacrebleu

BUDGETS=-1

STARTITER=0

export TF_USE_LEGACY_KERAS=1

while getopts d:m:p:l:s:e:k:n:i:v:a:bru:t:z:w:o:h:c:f: option
do
  case $option in
    d)
        DOMAIN=${OPTARG};;
    m)
        MODEL=${OPTARG};;
    p)
        PROMPT=${OPTARG};;
    l)
        NLINES=${OPTARG};;
    s)
        NSAMPLES=${OPTARG};;
    f)
        TEMPERATURE=${OPTARG};;
    e)
        EPS=${OPTARG};;
    k)
        TOPK=${OPTARG};;
    n)
        # Nucleus sampling
        TOPP=${OPTARG};;
    i)
        # TODO: Long options
        SIM=${OPTARG};;
    v)
        EVAL=${OPTARG};;
    a)
        # TODO: Enable arguments for algorithm e.g. k, div_pen
        ALGORITHM=${OPTARG};;
    b)
        DEBUG=1;;
    r)
        RECOMPUTE="--recompute_matrix";;
    u)
        # APPROXIMATION
        BUDGETS=${OPTARG};;
    o)
        # APPROXIMATION
        RZERO=${OPTARG};;
    h)
        # APPROXIMATION
        PALPHA=${OPTARG};;
    t)
        # DIVERSITY
        DOSAMPLE=0
        DIVERSITY=${OPTARG};;
    z)
        # DIVERSITY
        DIVERSEK=${OPTARG};;
    w)
        # DIVERSITY
        PAIRWISE=${OPTARG};;
    c)
        STARTITER=${OPTARG};;
    \?)
      echo "This is unexpected option." 1>&2
      exit 1
  esac
done

if [ "$ALGORITHM" == "beam" ]; then
    DOSAMPLE=-1
    DIVERSITY=0
elif [ "$ALGORITHM" == "dbs" ]; then
    DOSAMPLE=0
fi

DATADIR=None

# Return an error if the python script fails
set -e

MODELNAME=$(basename $MODEL)

mkdir -p ./results

python3 mbr/mbr_engine.py $DOMAIN \
    --model $MODELNAME \
    --sample_dir ./samples/$DOMAIN/$MODELNAME \
    --matrix_dir ./matrix/$DOMAIN/$MODELNAME \
    --n_lines $NLINES --start_iter $STARTITER \
    --n_samples $NSAMPLES \
    --temperature $TEMPERATURE \
    --eps $EPS --topk $TOPK --topp $TOPP \
    --sim $SIM \
    --eval $EVAL \
    --algorithm $ALGORITHM \
    $RECOMPUTE \
    --do_sample $DOSAMPLE --diversity_penalty $DIVERSITY \
    --diverse_k $DIVERSEK \
    --approx_budgets $BUDGETS \
    --pairwise_eval $PAIRWISE \
    --r_0 $RZERO --pruning_alpha $PALPHA


echo "done!"
