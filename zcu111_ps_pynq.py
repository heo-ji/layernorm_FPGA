"""
ZCU111 PS 서버 — LayerNorm HIL
===============================
실행 위치 : ZCU111 PS (PYNQ Linux)
실행 방법 : python3 zcu111_ps_pynq.py

역할
  1. PYNQ Overlay 로드 (bitstream + hwh)
  2. AXI-Lite 레지스터 초기화 (d_model, shift, mult)
  3. TCP 서버로 Host PC 요청 수신
  4. module_num 단위로 분할 → DMA → PL LayerNorm IP → 결과 반환

프로토콜 (Host ↔ PS)
  송신 (Host → PS) : [seq_len uint16 LE][d_model uint16 LE] + seq_len*d_model*int16 (row-major)
  수신 (PS → Host) : seq_len*d_model*int16 (row-major)

  * column-first 변환은 PS 내부에서 처리 (Host는 row-major 그대로 보내면 됨)
"""

import socket
import struct
import numpy as np
from pynq import Overlay, allocate

# ─────────────────────────────────────────────
# 설정값 (보드·모델에 맞게 수정)
# ─────────────────────────────────────────────
BIT_PATH    = "/home/xilinx/layernorm/layernorm_HIL.bit"
HOST        = "0.0.0.0"          # 보드의 모든 인터페이스에서 수신 (또는 "166.104.140.13")
PORT        = 5000
MODULE_NUM  = 8        # 128-bit bus → 8 rows per beat (16-bit × 8)

# d_model 별 레지스터 설정 (README_HW.md 파라미터 표 참조)
# reg0x08 = d_model
# reg0x0C = { shift_value[12:8] , mult_value[7:0] }
#         = (shift_value << 8) | mult_value

# x / d_model ≈ (x >>> shift_value) × (mult_value / 256)`


DMODEL_CONFIG = {
    128:  (128,  0x0540),   # shift=5,  mult=0x40  (BERT-tiny)
    192:  (192,  0x0655),   # shift=6,  mult=0x55
    384:  (384,  0x0755),   # shift=7,  mult=0x55
    768:  (768,  0x0855),   # shift=8,  mult=0x55  (BERT-base)
    1024: (1024, 0x0A40),   # shift=10, mult=0x40
}

# ─────────────────────────────────────────────
# PYNQ Overlay 로드 (서버 시작 시 1회)
# ─────────────────────────────────────────────
print(f"[PS] Overlay 로드 중: {BIT_PATH}")
ol    = Overlay(BIT_PATH)
dma   = ol.axi_dma_0
ln_ip = ol.layernorm_axi_wrapper_0
print("[PS] Overlay 로드 완료")
print(f"[PS] IP 목록: {list(ol.ip_dict.keys())}")


def init_registers(d_model: int):
    """d_model에 맞는 레지스터 값 설정."""
    if d_model not in DMODEL_CONFIG:
        raise ValueError(f"d_model={d_model} 은 지원하지 않음. 지원: {list(DMODEL_CONFIG.keys())}")
    reg_dmodel, reg_shift_mult = DMODEL_CONFIG[d_model]
    ln_ip.write(0x08, reg_dmodel)       # d_model
    ln_ip.write(0x0C, reg_shift_mult)   # shift | mult
    print(f"[PS] 레지스터 초기화: d_model={reg_dmodel}, shift_mult=0x{reg_shift_mult:04X}")


def recv_exact(conn, n: int) -> bytes:
    """소켓에서 정확히 n 바이트 수신."""
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Host 연결이 끊어짐")
        buf += chunk
    return buf


def run_layernorm_hw(data_int16: np.ndarray, d_model: int) -> np.ndarray:
    """
    data_int16 : (seq_len, d_model)  int16, row-major
    return     : (seq_len, d_model)  int16, row-major

    내부 동작:
      seq_len을 MODULE_NUM 단위로 분할
      각 chunk: (MODULE_NUM, d_model) → column-first → DMA ITER1 → ITER2 → 결과
    """
    seq_len = data_int16.shape[0]
    output  = np.zeros_like(data_int16)

    n_buf = MODULE_NUM * d_model   # DMA 버퍼 크기 (int16 개수)

    # DMA 버퍼는 한 번만 할당하고 반복 사용
    in_buf  = allocate(shape=(n_buf,), dtype=np.int16)
    out_buf = allocate(shape=(n_buf,), dtype=np.int16)

    try:
        n_chunks = (seq_len + MODULE_NUM - 1) // MODULE_NUM   # ceiling division

        for i in range(n_chunks):
            row_s = i * MODULE_NUM
            row_e = min(row_s + MODULE_NUM, seq_len)
            chunk = data_int16[row_s:row_e].copy()   # (actual_rows, d_model)

            # MODULE_NUM 미만이면 0 패딩
            actual_rows = chunk.shape[0]
            if actual_rows < MODULE_NUM:
                pad   = np.zeros((MODULE_NUM - actual_rows, d_model), dtype=np.int16)
                chunk = np.vstack([chunk, pad])

            # (MODULE_NUM, d_model) → column-first 직렬화
            flat = chunk.T.flatten()   # shape: (MODULE_NUM * d_model,)
            in_buf[:] = flat

            # ── ITER1: mean / invsqrt 계산 ──────────────────────
            ln_ip.write(0x00, 1)                    # cmd = iter1_start
            dma.sendchannel.transfer(in_buf)        # MM2S: PS → PL
            dma.sendchannel.wait()
            while ln_ip.read(0x04) != 1:            # status = WAIT_ITER2 polling
                pass

            # ── ITER2: normalization 계산 ───────────────────────
            ln_ip.write(0x00, 2)                    # cmd = iter2_start
            dma.sendchannel.transfer(in_buf)        # MM2S: 동일 입력 재전송
            dma.recvchannel.transfer(out_buf)       # S2MM: PL → PS
            dma.sendchannel.wait()
            dma.recvchannel.wait()

            # column-first → row-major 복원
            out_chunk = np.array(out_buf).reshape(d_model, MODULE_NUM).T
            # (MODULE_NUM, d_model) → 패딩 제거 후 저장
            output[row_s:row_e] = out_chunk[:actual_rows]

    finally:
        in_buf.freebuffer()
        out_buf.freebuffer()

    return output


# ─────────────────────────────────────────────
# TCP 서버 메인 루프
# ─────────────────────────────────────────────
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"[PS] TCP 서버 대기 중: {HOST}:{PORT}")

current_dmodel = None   # 레지스터 재설정 최소화

while True:
    conn, addr = server.accept()
    print(f"[PS] Host 연결: {addr}")

    try:
        while True:
            # ── 헤더 수신 ──
            try:
                header = recv_exact(conn, 4)
            except ConnectionError:
                break   # Host가 연결 종료 → 다음 accept 대기

            seq_len, d_model = struct.unpack('<HH', header)
            print(f"[PS] 수신 요청: seq_len={seq_len}, d_model={d_model}")

            # d_model 바뀌면 레지스터 재설정
            if d_model != current_dmodel:
                init_registers(d_model)
                current_dmodel = d_model

            # ── 데이터 수신 ──
            raw = recv_exact(conn, seq_len * d_model * 2)
            data_int16 = np.frombuffer(raw, dtype=np.int16).reshape(seq_len, d_model)

            # ── HW LayerNorm 실행 ──
            result_int16 = run_layernorm_hw(data_int16, d_model)

            # ── 결과 반환 ──
            conn.sendall(result_int16.tobytes())
            print(f"[PS] 결과 반환 완료: {result_int16.shape}")

    except Exception as e:
        print(f"[PS] 오류: {e}")
    finally:
        conn.close()
        print(f"[PS] Host 연결 종료: {addr}")
