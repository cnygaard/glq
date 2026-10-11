import json, collections, re
from safetensors import safe_open
D = "/opt/dlami/nvme/fn_mtp3"
qc = json.load(open(f"{D}/config.json"))["quantization_config"]
lb = qc["layer_bpw"]
print("=== quantization_config scalars ===")
print({k: v for k, v in qc.items() if not isinstance(v, (dict, list))})
print()
mtp = {k: v for k, v in lb.items() if k.startswith("mtp")}
ple = [k for k in lb if k.endswith("ngram_embedding")]
dec = [k for k in lb if k.startswith("model.")]
print("=== layer_bpw ===")
print(f"  total entries   : {len(lb)}")
print(f"  MTP entries     : {len(mtp)}  rungs={sorted(set(mtp.values()))}")
print(f"  PLE entry       : {lb[ple[0]] if ple else None} bpw")
print(f"  decoder entries : {len(dec)}  rungs={sorted(set(lb[k] for k in dec))}")
print()
f = safe_open(f"{D}/model.safetensors", framework="pt")
keys = set(f.keys())
kinds = collections.Counter()
for k in keys:
    if k.startswith("mtp"):
        g = re.sub(r"\.\d+\.", ".{n}.", k).replace("layers.0", "layers.N")
        kinds[g] += 1
print("=== MTP artifact tensor kinds in the output ===")
for k, v in sorted(kinds.items()):
    if any(a in k for a in ("trellis_packed", "SU", "SV", "Wscale")):
        print(f"  {v:6d}  {k}")
print()
print("=== stacked bf16 originals must be gone ===")
for bad in ("mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj"):
    print(f"  {bad.split('experts.')[1]:16s} {'STILL PRESENT (bug)' if bad in keys else 'superseded OK'}")
print()
print("=== norms/router must still be plain bf16 weights ===")
for keep in ("mtp.layers.0.self_attn.q_norm.weight",
             "mtp.layers.0.mlp.gate.weight",
             "mtp.layers.0.mlp.shared_expert_gate.weight",
             "mtp.pre_fc_norm_hidden.weight"):
    print(f"  {keep.replace('mtp.layers.0.','').replace('mtp.',''):34s} {'present OK' if keep in keys else 'MISSING (bug)'}")
