import itertools
import json
import os

# Định nghĩa các giá trị cần hoán vị
pos_values = [-20, 20, 50, 80]
neg_values = [-20, 20, 50, 80]
lp_values = [20, 0]

output_dir = "biases"
os.makedirs(output_dir, exist_ok=True)

# Đọc file bias mẫu (ví dụ: biases/test.bias) để lấy khung chuẩn
template_path = "biases/test.bias"
base_bias = {
    "bias_diff": 0,
    "bias_diff_off": 0,
    "bias_diff_on": 0,
    "bias_fo": 22,
    "bias_hpf": 0,
    "bias_refr": 0,
}

if os.path.exists(template_path):
  try:
    with open(template_path, "r") as f:
      base_bias = json.load(f)
  except Exception:
    pass  # Dùng khung mặc định nếu định dạng khác

# Tạo tất cả các tổ hợp (permutations / product)
count = 0
for pos, neg, lp in itertools.product(pos_values, neg_values, lp_values):
  # Tạo bản sao và gán giá trị mới
  current_bias = base_bias.copy()
  current_bias["bias_diff_on"] = pos
  current_bias["bias_diff_off"] = neg
  # Gán lowpass vào bias_hpf (hoặc bias_fo tùy cấu trúc sensor)
  current_bias["bias_hpf"] = lp

  filename = f"pos_{pos}_neg_{neg}_lp_{lp}.bias"
  filepath = os.path.join(output_dir, filename)

  with open(filepath, "w") as f:
    json.dump(current_bias, f, indent=4)

  count += 1

print(f"=== Đã tạo thành công {count} file bias tại thư mục: {output_dir}/ ===")
