DOMAIN=librispeech-asr-test
MODEL=openai/whisper-tiny #openai/whisper-large-v3
PROMPT=None
NLINES=2
STARTITER=0
NSAMPLES=4

TEMPERATURE=1.0
EPS=0.01
TOPK=0
TOPP=1.0 # nucleus
DOSAMPLE=1
DIVERSITY=1.0

MAXNEWTOKEN=-1

BSZ=16

QUANTIZE=-1
DISCARDPROB=""

export TF_USE_LEGACY_KERAS=1

while getopts d:m:p:l:s:f:e:k:n:t:b:z:a:q:x:r option
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
        TOPP=${OPTARG};;
    t)
        DOSAMPLE=0
        DIVERSITY=${OPTARG};;
    b)
        DOSAMPLE=-1
        DIVERSITY=0
        BSZ=${OPTARG};;
    z)
        BSZ=${OPTARG};;
    a)
        STARTITER=${OPTARG};;
    q)
        QUANTIZE=${OPTARG};;
    x)
        MAXNEWTOKEN=${OPTARG};;
    r)
        DISCARDPROB="--discard_prob";;
    \?)
      echo "This is unexpected option." 1>&2
      exit 1
  esac
done

# TODO:
MINBSZ=$(( $BSZ < $NSAMPLES ? $BSZ : $NSAMPLES ))
BSZ=$MINBSZ

if [ "$DOMAIN" == "xsum" ]; then
    DATADIR=xsum
elif [ "$DOMAIN" == "cnndm" ]; then
    DATADIR=cnn_cln
elif [ "$DOMAIN" == "samsum" ]; then
    DATADIR=None
elif [[ $DOMAIN == "wmt21"* ]]; then
    # https://stackoverflow.com/questions/229551/how-to-check-if-a-string-contains-a-substring-in-bash
    DATADIR=wmt21
elif [[ $DOMAIN == "wmt19"* ]]; then
    DATADIR=wmt19-text
elif [[ $DOMAIN == "iwslt"* ]]; then
    DATADIR=iwslt17-text
elif [ "$DOMAIN" == "nocaps" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "mscoco-ft" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "squad_v2" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "common_gen" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "e2e_nlg" ]; then
    DATADIR=None
else
    echo "No cache available for dataset: $DOMAIN. Downloading from huggingface datasets."
    DATADIR=None
fi

# TODO: Check parameters here

if [ "$DATADIR" == "None" ]; then
    echo "uses huggingface dataset"
else
    echo "downloading $DATADIR"
    python3 experiments/download.py $DATADIR

    mkdir -p ./dataset/$DATADIR

    cp ./fairseq/$DATADIR/* ./dataset/$DATADIR/
fi


echo "sampling..."
python3 mbr/sample.py $DOMAIN \
    --model $MODEL --prompt $PROMPT \
    --n_lines $NLINES --start_iter $STARTITER \
    --n_samples $NSAMPLES \
    --temperature $TEMPERATURE \
    --eps $EPS --topk $TOPK --topp $TOPP \
    --do_sample $DOSAMPLE --diversity_penalty $DIVERSITY \
    --bsz $BSZ \
    --quantize $QUANTIZE \
    --max_new_token $MAXNEWTOKEN \
    $DISCARDPROB

