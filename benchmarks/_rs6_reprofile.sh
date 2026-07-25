#!/bin/bash
# RS6: post-RS1-RS4a re-profile of the 3inst-kernel checkpoint. Sequential stages,
# ONE GPU job at a time. Expected vs Phase-0: input_rht ~0, output RHT shrunk,
# matvec ~40-43%, B=1 129 -> ~160-175 tok/s, PPL == 9.2299, matvec regs <= 64 no spill.
set -u
export PATH=/usr/local/cuda/bin:$PATH
cd /opt/dlami/nvme/golay-leech-quant
source /home/ubuntu/venv/bin/activate
CK=/opt/dlami/nvme/ck3inst
PORT=8321
BENCH_COMMON="--base-url http://localhost:$PORT --model $CK --dataset-name random \
  --random-input-len 128 --random-output-len 256 --ignore-eos"

wait_health() {  # arg: server log for failure dump
    for _ in $(seq 1 90); do
        curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 5
    done
    echo "RESULT serve FAILED"; tail -30 "$1"; return 1
}

echo "=== RS6 stage 0: editable install (engine subprocess must get repo kernels) ==="
pip uninstall -y glq >/dev/null 2>&1
pip install -e . --no-deps -q 2>&1 | tail -2
python -c "import glq; print('RESULT glq_file', glq.__version__, glq.__file__)"

echo "=== RS6 stage 1: checkpoint download ==="
python -c "from huggingface_hub import snapshot_download as d; \
print('RESULT ckpt', d('xv0y5ncu/SmolLM3-3B-trellis-3inst-4bpw-kernel', local_dir='$CK'))"

echo "=== RS6 stage 2: PPL gate (expect 9.2299) ==="
stdbuf -oL -eL python benchmarks/_ppl_checkpoint.py --model $CK 2>&1 | tail -4

echo "=== RS6 stage 3: clean serve + bench ==="
vllm serve $CK --quantization glq --max-model-len 2048 --port $PORT \
    >/opt/dlami/nvme/rs6_serve_clean.log 2>&1 &
SRV=$!
wait_health /opt/dlami/nvme/rs6_serve_clean.log || exit 1
echo "--- RESULT bench B=1 ---"
vllm bench serve $BENCH_COMMON --num-prompts 16 --max-concurrency 1 2>&1 \
    | grep -E "Output token throughput|Mean TPOT|Median TPOT|Benchmark duration"
echo "--- RESULT bench B=32 ---"
vllm bench serve $BENCH_COMMON --num-prompts 128 --max-concurrency 32 2>&1 \
    | grep -E "Output token throughput|Mean TPOT|Median TPOT|Benchmark duration"
kill $SRV; wait $SRV 2>/dev/null

# Scoped nsys via INTERACTIVE SESSION mode — the only recipe that works on vLLM 0.25
# (capture-range=cudaProfilerApi never fires from the engine-core subprocess, and
# --duration expiry SIGTERMs the server). --cuda-graph-trace=node is mandatory under
# FULL_AND_PIECEWISE or the kernel table comes back empty.
for CONC in 1 32; do
    echo "=== RS6 stage 4/$CONC: scoped nsys B=$CONC ==="
    NP=$((CONC == 1 ? 200 : 600))
    SESS=rs6b$CONC
    nsys launch --session-new=$SESS --trace=cuda --cuda-graph-trace=node \
        vllm serve $CK --quantization glq --max-model-len 2048 --port $PORT \
        >/opt/dlami/nvme/rs6_serve_nsys$CONC.log 2>&1 &
    wait_health /opt/dlami/nvme/rs6_serve_nsys$CONC.log || exit 1
    timeout 300 vllm bench serve $BENCH_COMMON --num-prompts $NP --max-concurrency $CONC \
        >/opt/dlami/nvme/rs6_bench$CONC.log 2>&1 &
    BENCH=$!
    for _ in $(seq 1 60); do    # capture only once decode is live
        tail -5 /opt/dlami/nvme/rs6_serve_nsys$CONC.log \
            | grep -qE "generation throughput: [1-9]" && break
        sleep 3
    done
    nsys start --session=$SESS -o /opt/dlami/nvme/rs6_b$CONC --force-overwrite=true
    sleep 20
    nsys stop --session=$SESS
    wait $BENCH 2>/dev/null
    grep -E "Output token throughput|Mean TPOT" /opt/dlami/nvme/rs6_bench$CONC.log
    SRV=$(pgrep -f "vllm serve $CK" | head -1); [ -n "$SRV" ] && kill $SRV
    sleep 5
    echo "--- RESULT nsys kern table B=$CONC ---"
    nsys stats --report cuda_gpu_kern_sum --force-export=true --format table \
        /opt/dlami/nvme/rs6_b$CONC.nsys-rep 2>&1 | grep -E "^\||glq|flash|gemv" | head -45
done

echo "=== RS6 stage 5: ncu fused matvec (regs <= 64, local traffic 0) ==="
sudo -E env "PATH=$PATH" /usr/local/cuda/bin/ncu \
    --kernel-name "regex:glq_trellis_matvec" --launch-count 6 \
    --metrics launch__registers_per_thread,launch__thread_count,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum \
    /home/ubuntu/venv/bin/python benchmarks/_rs6_ncu_matvec.py 2>&1 \
    | grep -E "RESULT|glq_trellis_matvec|registers_per_thread|local_op|thread_count" | head -40

echo "=== RS6 DONE ==="
