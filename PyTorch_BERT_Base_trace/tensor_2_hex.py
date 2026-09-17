"""
PyTorch_BERT_Base_trace/tensor_2_hex.py

custom_norm.py의 forward_fxp88에서 저장한 실제 BERT 활성값(.pt)을 전부 훑어서,
RTL testbench(tb_top_calculator_one_row.v)가 $readmemh로 이어서(back-to-back) 읽을
hex trace 파일로 변환한다.

입력 텐서 shape: (BATCH, SEQ_LEN, D_MODEL) = (8, 128, 768), Q8.8로 floor+clip 된
float 텐서 (forward_fxp88의 input_fx16 저장분 등).

row 매핑 (README_HW.md: "row = 토큰"):
  한 문장(batch index b) 안에서 SEQ_LEN을 parallel_row_num개씩 잘라
  "연속된 토큰들"을 한 그룹의 row로 쓴다.
    group g (0..SEQ_LEN/P-1) = tensor[b, g*P:(g+1)*P, :]  ->  (P, D_MODEL)
  그룹 하나 = trace 768(D_MODEL)줄. 한 .pt 파일의 모든 (batch, group)을
  순서대로 이어붙여 파일 하나로 출력 -> tb가 처음부터 끝까지 쭉 읽으며
  그룹마다 IDLE->CALC_ACCUM->...->DONE->IDLE 을 반복하게 됨.

출력 포맷: 한 줄 = 그 cycle의 P개 row를 패킹한 hex (row[r]가
bits[W*(r+1)-1 : W*r] 차지, W=신호별 비트폭, gen_trace128.py와 동일 규약).
신호별 비트폭/소수점 위치(W, frac_bits)는 코드 상단 FORMAT_TABLE에서 정의 —
새 종류의 텐서가 생기면 파싱 옵션이 아니라 그 표에 한 줄 추가하면 됨.
용량을 줄이려고 앞자리 0은 채우지 않음 ($readmemh는 짧은 hex를 자동 0-확장하므로 안전).

폴더 안의 .pt를 전부 자동으로 훑는다 (지금은 layer0/5/11 등 일부만 있지만
같은 이름 규칙으로 다른 레이어가 생겨도 그대로 처리됨).

출력 위치: --tensor_dir(예: mnli/tensor)의 부모 폴더(mnli, task별로 달라짐) 밑에
  hex_txt_{parallel_row_num}x{d_model}/  (예: hex_txt_32x768)
폴더명의 d_model은 실제 .pt 중 하나(input/normalized처럼 마지막 축이 1이 아닌 것)의
shape[2]를 읽어서 정함 — task/레이어가 바뀌어도 자동으로 맞는 폴더명이 나옴.

사용 예:
  python tensor_2_hex.py                         # mnli/tensor/*.pt 전부 -> mnli/hex_txt_32x768/
  python tensor_2_hex.py --tensor_dir mnli/tensor --parallel_row_num 32
  python tensor_2_hex.py --tensor_dir mnli/tensor --device cuda  # CUDA가 보이는 환경

profiling_pass2(.pt의 dim0 = forward K개를 이어붙인 K*batch_size 문장)로 뽑은 경우:
  --num_forwards로 앞에서부터 K개 forward만 잘라 K별 폴더로 따로 출력한다.
  forward #0~K-1은 K가 커져도 앞부분이 같으므로 pass2는 가장 큰 K로 한 번만 돌리면 됨.
  python tensor_2_hex.py --tensor_dir mnli/tensor --num_forwards 1 2 5 10
    -> mnli/hex_txt_32x768_k1/, _k2/, _k5/, _k10/
  (TB는 -generic_top "NUM_FORWARDS=K"로 같은 K 폴더를 읽음)
"""
import argparse
import ctypes
import glob
import os

import torch

# ── 신호별 비트폭/포맷 (doc/README_HW.md §5 "IP 파라미터(2)" 표 그대로) ──────
# {.pt 파일명 끝 접미사(_{kind}.pt): (total_bits, frac_bits)}
# tensor_2_hex.py가 실제로 다루는 건 input/mean/invsqrt/normalized 4가지뿐이고
# 넷 다 16bit 8.8로 우연이 아니라 전부 top_calculator_one_row/top_normalization의
# 최종 입출력 포트 폭(DATA_WIDTH=16)에 맞춰진 것들이라 그렇다.
# accum/squared_accum/mean2/var는 모듈 내부 중간값이라 .pt로 안 나오므로 안 씀 —
# 나중에 그것도 저장해서 여기 넣게 되면 표 값 그대로 추가하면 됨.
FORMAT_TABLE = {
    "input":            (16, 8),   # DATA_WIDTH=16,           8.8
    "mean":             (16, 8),   # MEAN_DATA_WIDTH=16,      8.8
    "invsqrt":          (16, 8),   # LUT_NUM=24 출력,         8.8
    "normalized":       (16, 8),   # output,                  8.8
    # 참고용(현재 미사용): accum=(26,8) 18.8 / squared_accum=(34,8) 26.8 /
    #                      squared_mean=(32,16) 16.16 / var=(24,16) 8.16
}


def format_for(base_name: str):
    """'layer0_atten_input' -> FORMAT_TABLE['input'] 처럼 파일명 끝 접미사로 포맷 결정."""
    kind = base_name.rsplit("_", 1)[-1]
    if kind not in FORMAT_TABLE:
        raise ValueError(f"'{base_name}'의 종류('{kind}')가 FORMAT_TABLE에 없음 — "
                          f"코드 상단 FORMAT_TABLE에 추가할 것")
    return FORMAT_TABLE[kind]


def _write_tensor_bytes(f, data):
    """연속된 CPU uint8 텐서를 Python 원소 변환 없이 파일에 쓴다."""
    data = data.contiguous().cpu()
    f.write(ctypes.string_at(data.data_ptr(), data.numel()))


def _dump_hex_rows(values, out_path, bits, frac_bits, chunk_lines):
    """values[N, lanes]를 벡터 연산으로 양자화/hex 인코딩해 줄 단위로 기록한다.

    lanes > 1이면 마지막 lane부터 이어 붙인다. 이는 기존 pack_rows()에서 lane 0을
    LSB에 놓고 정수 전체를 %X로 출력하던 것과 같은 순서다.
    """
    if values.dim() != 2:
        raise ValueError(f"내부 오류: values는 2D여야 함 (shape={tuple(values.shape)})")
    if values.shape[1] > 1 and bits % 4:
        raise ValueError("여러 row를 묶는 빠른 변환은 비트폭이 4의 배수여야 함")

    device = values.device
    hex_digits = (bits + 3) // 4
    shifts = torch.arange(hex_digits - 1, -1, -1, device=device, dtype=torch.int64) * 4
    hex_ascii = torch.tensor(bytearray(b"0123456789ABCDEF"), device=device, dtype=torch.uint8)
    mask = (1 << bits) - 1
    scale = 1 << frac_bits

    with open(out_path, "wb") as f, torch.inference_mode():
        for start in range(0, values.shape[0], chunk_lines):
            chunk = values[start:start + chunk_lines]
            quantized = torch.floor(chunk * scale).to(torch.int64).bitwise_and_(mask)
            if quantized.shape[1] > 1:
                quantized = quantized.flip(1)

            # [line, lane, hex digit] -> [line, packed hex digit]
            digits = ((quantized.unsqueeze(-1) >> shifts) & 0xF).reshape(quantized.shape[0], -1)
            width = digits.shape[1]

            # 기존 %X 출력과 똑같이 줄 앞쪽의 0만 제거한다(값 0은 "0" 유지).
            nonzero = digits.ne(0)
            first = nonzero.to(torch.int8).argmax(dim=1)
            first = torch.where(nonzero.any(dim=1), first, width - 1)
            lengths = width - first + 1  # 마지막 newline 포함
            offsets = lengths.cumsum(0) - lengths
            total_bytes = int(lengths.sum().item())

            out = torch.empty(total_bytes, device=device, dtype=torch.uint8)
            columns = torch.arange(width, device=device)
            keep = columns.unsqueeze(0) >= first.unsqueeze(1)
            destinations = offsets.unsqueeze(1) + columns.unsqueeze(0) - first.unsqueeze(1)
            out[destinations[keep]] = hex_ascii[digits[keep]]
            out[offsets + lengths - 1] = ord("\n")
            _write_tensor_bytes(f, out)


def dump_vector_tensor(tensor, parallel_row_num, out_path, bits, frac_bits, chunk_lines=4096):
    """(BATCH, SEQ_LEN, D_MODEL) 텐서 -> 문장별로 SEQ_LEN을 parallel_row_num개씩
    잘라 그룹마다 D_MODEL줄, 전체 그룹을 이어붙여 파일 하나로 씀."""
    batch, seq_len, d_model = tensor.shape
    groups_per_batch = seq_len // parallel_row_num
    if seq_len % parallel_row_num != 0:
        print(f"  [warn] seq_len={seq_len} 이 parallel_row_num={parallel_row_num} 로 "
              f"안 나눠떨어져서 뒤 {seq_len % parallel_row_num}개 토큰은 버림")

    usable_seq_len = groups_per_batch * parallel_row_num
    # 출력 순서: batch -> group -> d_model -> lane
    values = (tensor[:, :usable_seq_len, :]
              .reshape(batch, groups_per_batch, parallel_row_num, d_model)
              .permute(0, 1, 3, 2)
              .reshape(-1, parallel_row_num))
    _dump_hex_rows(values, out_path, bits, frac_bits, chunk_lines)
    return values.shape[0], batch * groups_per_batch


def dump_scalar_tensor(tensor, parallel_row_num, out_path, bits, frac_bits, chunk_lines=4096):
    """(BATCH, SEQ_LEN, 1) 텐서(mean/invsqrt) -> 위와 동일한 (batch, group) 순서로
    row당 한 줄씩(스칼라) 이어붙여 씀 (golden 비교용)."""
    batch, seq_len, _ = tensor.shape
    groups_per_batch = seq_len // parallel_row_num

    usable_seq_len = groups_per_batch * parallel_row_num
    values = tensor[:, :usable_seq_len, 0].reshape(-1, 1)
    _dump_hex_rows(values, out_path, bits, frac_bits, chunk_lines)
    return values.shape[0], batch * groups_per_batch


def load_tensor(path):
    """텐서 trace만 안전 모드로 로드한다."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # weights_only가 없는 구버전 PyTorch 호환
        return torch.load(path, map_location="cpu")


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device에 CUDA를 지정했지만 torch.cuda.is_available()이 False임")
    return device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_dir", type=str,
                     default=os.path.join(os.path.dirname(__file__), "mnli/tensor"),
                     help=".pt 파일들이 있는 폴더 (하위 폴더까지 재귀적으로 훑음)")
    ap.add_argument("--parallel_row_num", type=int, default=32,
                     help="128BUS 프로젝트=8, 512BUS 프로젝트=32")
    ap.add_argument("--num_forwards", type=int, nargs="+", default=None,
                     help="profiling_pass2 .pt에서 앞 K개 forward만 잘라 hex_txt_{P}x{D}_k{K}/로 출력 "
                          "(예: --num_forwards 1 2 5 10). 안 주면 .pt 전체를 hex_txt_{P}x{D}/로 출력")
    ap.add_argument("--batch_size", type=int, default=8,
                     help="pass2를 돌릴 때의 per_device_eval_batch_size (forward 1개 = batch_size 문장)")
    ap.add_argument("--device", type=str, default="auto",
                    help="변환 연산 장치: auto, cpu, cuda, cuda:0 등 (기본값: auto)")
    ap.add_argument("--chunk_lines", type=int, default=4096,
                    help="한 번에 hex로 인코딩할 출력 줄 수 (GPU/CPU 메모리가 부족하면 낮출 것)")
    args = ap.parse_args()

    if args.chunk_lines < 1:
        ap.error("--chunk_lines는 1 이상이어야 함")
    device = resolve_device(args.device)
    print(f"[tensor_2_hex] device = {device}")

    pt_files = sorted(glob.glob(os.path.join(args.tensor_dir, "**", "*.pt"), recursive=True))
    if not pt_files:
        print(f"[tensor_2_hex] {args.tensor_dir} 안에 .pt 파일이 없음")
        return

    d_model = None
    for pt_path in pt_files:
        t = load_tensor(pt_path)
        if t.dim() == 3 and t.shape[2] > 1:
            d_model = t.shape[2]
            break
    if d_model is None:
        raise ValueError("d_model을 결정할 3D(마지막 축>1) 텐서(input/normalized류)를 못 찾음")
    del t

    task_dir = os.path.dirname(os.path.normpath(args.tensor_dir))  # 예: mnli/tensor -> mnli

    output_dirs = {}
    for k in (args.num_forwards or [None]):
        suffix = "" if k is None else f"_k{k}"
        out_dir = os.path.join(task_dir, f"hex_txt_{args.parallel_row_num}x{d_model}{suffix}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[tensor_2_hex] out_dir = {out_dir}")
        output_dirs[k] = out_dir

    # 1.3GB 안팎의 trace 전체를 RAM에 쌓지 않고 파일 하나씩 변환한다.
    for pt_path in pt_files:
        t_cpu = load_tensor(pt_path)
        if t_cpu.dim() != 3:
            print(f"  [skip] {pt_path}: 3D 텐서가 아님 (shape={tuple(t_cpu.shape)})")
            continue
        t_full = t_cpu.to(device)

        for k, out_dir in output_dirs.items():
            t = t_full
            if k is not None:
                # dim0 = [forward0의 batch_size문장, forward1의 ...] 순서라 앞에서 자르면 forward #0~k-1
                n_sentences = k * args.batch_size
                if t.shape[0] < n_sentences:
                    print(f"  [skip] {pt_path}: 문장 {t.shape[0]}개 < num_forwards {k} x batch_size {args.batch_size}")
                    continue
                t = t[:n_sentences]
            batch, seq_len, last_dim = t.shape
            if batch < 1 or seq_len < args.parallel_row_num:
                print(f"  [skip] {pt_path}: seq_len={seq_len} < parallel_row_num={args.parallel_row_num}")
                continue

            base = os.path.splitext(os.path.basename(pt_path))[0]
            out_path = os.path.join(out_dir, f"{base}.txt")
            bits, frac_bits = format_for(base)

            if last_dim == 1:
                n_lines, n_groups = dump_scalar_tensor(
                    t, args.parallel_row_num, out_path, bits, frac_bits, args.chunk_lines)
            else:
                n_lines, n_groups = dump_vector_tensor(
                    t, args.parallel_row_num, out_path, bits, frac_bits, args.chunk_lines)

            size_kb = os.path.getsize(out_path) / 1024
            print(f"[tensor_2_hex] {pt_path} -> {out_path}  "
                  f"({n_groups} group x {args.parallel_row_num} row, {n_lines} line, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
