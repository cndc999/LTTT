"""
Nén ảnh bằng Huffman (lossless).
Pipeline: Ảnh -> bytes pixel -> chuỗi bit -> Huffman -> file .huff -> giải nén -> ảnh

Cài thư viện:  pip install pillow
Chạy:          python huffman_image.py anh.jpg
               python huffman_image.py anh.jpg --gray   (nén ảnh xám)
"""
import heapq
import os
import struct
import sys
from collections import Counter

from PIL import Image


# ---------- 1. Ảnh -> bytes -> chuỗi bit ----------
def image_to_bytes(path, gray=False):
    img = Image.open(path).convert("L" if gray else "RGB")
    return img.tobytes(), img.mode, img.size  # size = (width, height)


def bytes_to_bits(data):
    return "".join(f"{b:08b}" for b in data)


# ---------- 2. Huffman ----------
class Node:
    def __init__(self, freq, sym=None, left=None, right=None):
        self.freq, self.sym, self.left, self.right = freq, sym, left, right

    def __lt__(self, other):  # để heapq so sánh được
        return self.freq < other.freq


def build_tree(freq):
    heap = [Node(f, s) for s, f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:  # trường hợp chỉ có 1 symbol
        return Node(heap[0].freq, None, heap[0], None)
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, Node(a.freq + b.freq, None, a, b))
    return heap[0]


def build_codes(root):
    codes = {}
    stack = [(root, "")]
    while stack:  # duyệt bằng stack (tránh đệ quy sâu)
        node, code = stack.pop()
        if node.sym is not None:
            codes[node.sym] = code or "0"
        else:
            if node.right:
                stack.append((node.right, code + "1"))
            if node.left:
                stack.append((node.left, code + "0"))
    return codes


# ---------- 3. Nén / giải nén ----------
def compress(data, mode, size, out_path):
    freq = Counter(data)
    codes = build_codes(build_tree(freq))
    bits = "".join(codes[b] for b in data)
    nbits = len(bits)  # số bit sau mã hóa (chưa tính padding)

    pad = (8 - len(bits) % 8) % 8
    bits += "0" * pad
    payload = int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""

    with open(out_path, "wb") as f:
        f.write(b"HUF1")
        f.write(struct.pack(">B", 1 if mode == "L" else 3))      # số kênh
        f.write(struct.pack(">IIH", size[0], size[1], len(freq)))  # w, h, số symbol
        for sym, fr in freq.items():                              # bảng tần số
            f.write(struct.pack(">BI", sym, fr))
        f.write(struct.pack(">B", pad))
        f.write(payload)
    return nbits


def decompress(in_path, out_img_path):
    with open(in_path, "rb") as f:
        assert f.read(4) == b"HUF1", "File không hợp lệ"
        channels, w, h, n = struct.unpack(">BIIH", f.read(1 + 4 + 4 + 2))
        freq = {}
        for _ in range(n):
            sym, fr = struct.unpack(">BI", f.read(5))
            freq[sym] = fr
        pad = struct.unpack(">B", f.read(1))[0]
        payload = f.read()

    root = build_tree(freq)
    total = w * h * channels
    bits = bin(int.from_bytes(payload, "big"))[2:].zfill(len(payload) * 8)
    if pad:
        bits = bits[:-pad]

    out, node = bytearray(), root
    for bit in bits:
        node = node.left if bit == "0" else node.right
        if node.sym is not None:
            out.append(node.sym)
            node = root
            if len(out) == total:
                break

    mode = "L" if channels == 1 else "RGB"
    Image.frombytes(mode, (w, h), bytes(out)).save(out_img_path)
    return bytes(out)


# ---------- 4. Main ----------
def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python huffman_image.py <anh> [--gray]")
        return
    path, gray = sys.argv[1], "--gray" in sys.argv
    base = os.path.splitext(path)[0]

    data, mode, size = image_to_bytes(path, gray)
    bits = bytes_to_bits(data)
    print(f"Kích thước ảnh     : {size[0]}x{size[1]} ({mode})")
    nbits = compress(data, mode, size, base + ".huff")
    comp_size = os.path.getsize(base + ".huff")

    print(f"Số bit ảnh trước mã hóa   : {len(bits):,} bit")
    print(f"Số bit sau mã hóa         : {nbits:,} bit")
    print(f"Kích thước trước mã hóa   : {len(data):,} byte")
    print(f"Kích thước sau mã hóa     : {comp_size:,} byte (file .huff, gồm header)")
    print(f"Hiệu suất nén             : {(1 - comp_size / len(data)) * 100:.2f}%")

    # Giải nén và lưu ảnh khôi phục (không in kiểm tra)
    decompress(base + ".huff", base + "_restored.png")


if __name__ == "__main__":
    main()
