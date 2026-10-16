import re
from stable_baselines3 import PPO


def extract_layers(model):
    sd = model.policy.state_dict()
    weights, biases = {}, {}

    def order(key):
        if "policy_net" in key:
            m = re.search(r"policy_net\.(\d+)", key)
            return int(m.group(1)) if m else 0
        return 10 ** 6      # action_net 排最后

    for key, val in sd.items():
        if not ("policy_net" in key or "action_net" in key):
            continue
        if key.endswith(".weight"):
            weights[order(key)] = val.detach().cpu().numpy()
        elif key.endswith(".bias"):
            biases[order(key)] = val.detach().cpu().numpy()

    return [{"W": weights[i], "b": biases[i]} for i in sorted(weights)]
def write_c_array(name, arr):
    if arr.ndim == 2:
        rows, cols = arr.shape
        lines = [f"static const float {name}[{rows}][{cols}] = {{"]
        for row in arr:
            lines.append("    {" + ", ".join(f"{x:.8f}" for x in row) + "},")
        lines.append("};\n")
        return "\n".join(lines)
    else:
        return (f"static const float {name}[{arr.shape[0]}] = {{\n"
                + ", ".join(f"{x:.8f}" for x in arr) + "\n};\n")


def generate_forward(layers):
    code = ["#include <math.h>",
            "static inline void forward_policy(const float input[], float output[]) {",
            "    const float *curr = input;"]
    for i, layer in enumerate(layers):
        W, b = layer["W"], layer["b"]
        out_dim, in_dim = W.shape
        code.append(f"    static float layer{i}[{out_dim}];")
        code.append(f"    for (int i = 0; i < {out_dim}; i++) {{")
        code.append(f"        float s = w{i}_bias[i];")
        code.append(f"        for (int j = 0; j < {in_dim}; j++)")
        code.append(f"            s += w{i}_weight[i][j] * curr[j];")
        if i < len(layers) - 1:
            code.append(f"        layer{i}[i] = tanhf(s);")   # 隐藏层 tanh
        else:
            code.append(f"        layer{i}[i] = s;")          # 输出层线性
        code.append("    }")
        code.append(f"    curr = layer{i};")
    final = layers[-1]["W"].shape[0]
    code.append(f"    for (int i = 0; i < {final}; i++) output[i] = curr[i];")
    code.append("}")
    return "\n".join(code)

def main():
    model = PPO.load("robo1_getup_ppo")
    layers = extract_layers(model)

    with open("policy_network.h", "w") as f:
        f.write("// Auto-generated policy-only MLP\n\n#pragma once\n\n")
        f.write(f"#define OBS_DIM {layers[0]['W'].shape[1]}\n")
        f.write(f"#define ACTION_DIM {layers[-1]['W'].shape[0]}\n\n")
        for i, layer in enumerate(layers):
            f.write(write_c_array(f"w{i}_weight", layer["W"]))
            f.write(write_c_array(f"w{i}_bias", layer["b"]))
        f.write(generate_forward(layers))
    print("Generated policy_network.h")


if __name__ == "__main__":
    main()