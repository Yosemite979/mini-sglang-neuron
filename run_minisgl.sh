unset NEURON_VISIBLE_DEVICES

export TP_SIZE=2
#export PJRT_DEVICE=NEURON
export NEURON_RT_NUM_CORES="${TP_SIZE}"
export LOGLEVEL=DEBUG
#export NEURON_RT_VISIBLE_CORES=0-1
echo "NEURON_VISIBLE_DEVICES=${NEURON_VISIBLE_DEVICES:-<unset>} NEURON_RT_NUM_CORES=${NEURON_RT_NUM_CORES:-<unset>} NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-<unset>}"
python -m minisgl \
  --model-path /root/data/Qwen/Qwen3-0.6B \
  --dtype bfloat16 \
  --tp-size "$TP_SIZE" \
  --max-running-requests 64 \
  --max-seq-len-override 127 \
  --num-pages 8192 \
  --port 1919 \
  --shell 2>&1 | tee /root/data/sgl.log
