DATA_ROOT=$1

export PYTHONPATH=.
TYPE=hubert
KM_MODEL_PATH=./ckpts/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
CKPT_PATH=./ckpts/mhubert_base_vp_en_es_fr_it3.pt
LAYER=11

MANIFEST=$DATA_ROOT/wav_16k/unit_manifest_train.txt
OUT_QUANTIZED_FILE==$DATA_ROOT/wav_16k/unit_train.txt
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".wav"


MANIFEST=$DATA_ROOT/wav_16k/unit_manifest_valid.txt
OUT_QUANTIZED_FILE==$DATA_ROOT/wav_16k/unit_valid.txt
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".wav"

MANIFEST=$DATA_ROOT/wav_16k/unit_manifest_test.txt
OUT_QUANTIZED_FILE==$DATA_ROOT/wav_16k/unit_test.txt
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".wav"