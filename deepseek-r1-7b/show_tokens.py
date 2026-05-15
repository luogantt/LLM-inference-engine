"""
解析 DeepSeek-R1-Distill-Qwen-7B tokenizer，正确解码所有中文 token。
GPT-2 风格 byte-level BPE: bytes 0-255 → Unicode 字符，需要逆向映射。
"""

import json
import re
from tokenizers import Tokenizer

TOK_PATH = "/home/lg/推理/推理引擎/deepseek-r1-7b/tokenizer.json"


def build_byte_decoder():
    """构建 GPT-2 bytes-to-unicode 逆向映射表"""
    # 与 GPT-2 / Qwen tokenizer 的 bytes_to_unicode() 一致
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)  # 超出 latin-1 的字节映射到 U+0100+
            n += 1
    cs = [chr(c) for c in cs]
    byte_to_char = dict(zip(bs, cs))
    # 逆向: char → byte
    char_to_byte = {c: b for b, c in byte_to_char.items()}
    return char_to_byte


CHAR_TO_BYTE = build_byte_decoder()


def token_str_to_text(token_str: str) -> str:
    """将 tokenizer 内部字符串 -> 原始 UTF-8 文本"""
    try:
        raw_bytes = bytes(CHAR_TO_BYTE.get(c, ord(c)) for c in token_str)
        return raw_bytes.decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return token_str


def has_cjk(s):
    for ch in s:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF):
            return True
    return False


def main():
    tok = Tokenizer.from_file(TOK_PATH)
    vocab = json.load(open(TOK_PATH))["model"]["vocab"]

    print(f"词表大小: {tok.get_vocab_size()}")
    print()

    # ── 1. 统计中文 token ──────────────────────────────────
    chinese = []
    for raw_str, tid in vocab.items():
        text = token_str_to_text(raw_str)
        if has_cjk(text):
            chinese.append((text, tid))

    single_char = [(t, i) for t, i in chinese if len(t) == 1]
    multi_char = [(t, i) for t, i in chinese if len(t) > 1]

    print(f"包含中文的 token: {len(chinese)} / {len(vocab)}")
    print(f"  单字 token:     {len(single_char)}")
    print(f"  多字/词语 token: {len(multi_char)}")
    print()

    # ── 2. 按 ID 排序展示 ──────────────────────────────────
    sorted_cn = sorted(chinese, key=lambda x: x[1])

    print("=" * 55)
    print("前 80 个中文 token (按 ID = 频率从高到低)")
    print("=" * 55)
    for t, tid in sorted_cn[:80]:
        print(f"  [{tid:6d}]  {t}")
    print(f"  ... (共 {len(sorted_cn)} 个)")
    print()

    # ── 3. 高频单字 ────────────────────────────────────────
    print("=" * 55)
    print("高频单字 (ID < 50000 = 训练语料中的高频字)")
    print("=" * 55)
    highfreq_single = [(t, i) for t, i in single_char if i < 50000]
    for t, tid in highfreq_single:
        print(f"  [{tid:6d}]  {t}")
    print(f"  (共 {len(highfreq_single)} 个)")
    print()

    # ── 4. 多字词语展示 ────────────────────────────────────
    print("=" * 55)
    print("多字词语 token 示例 (前 50 个, 按 ID)")
    print("=" * 55)
    multi_sorted = sorted(multi_char, key=lambda x: x[1])
    for t, tid in multi_sorted[:50]:
        print(f"  [{tid:6d}]  {t}")
    print(f"  ... (共 {len(multi_sorted)} 个)")
    print()

    # ── 5. 分词实例 ────────────────────────────────────────
    print("=" * 55)
    print("分词实例")
    print("=" * 55)
    samples = [
        "你好世界",
        "今天天气真好",
        "人工智能正在改变世界",
        "深度求索",
        "DeepSeek是由深度求索公司开发的大语言模型",
        "我喜欢学习大模型推理",
        "今天是2024年1月1日",
        "循环神经网络和Transformer的区别是什么",
    ]
    for s in samples:
        enc = tok.encode(s)
        decoded_tokens = [token_str_to_text(t) for t in enc.tokens]
        print(f"  原文:   {s}")
        print(f"  Tokens: {decoded_tokens}")
        print(f"  IDs:    {enc.ids}")
        print()

    # ── 6. 特殊 Token ──────────────────────────────────────
    print("=" * 55)
    print("特殊 Token (ID 151643+)")
    print("=" * 55)
    for tid in range(151643, tok.get_vocab_size()):
        raw = tok.id_to_token(tid)
        if raw:
            text = token_str_to_text(raw)
            print(f"  [{tid}]  {text}")

    # ── 7. 数字/英文/中文混合 token ────────────────────────
    print()
    print("=" * 55)
    print("混合 token (中文+数字/英文)")
    print("=" * 55)
    mixed = [(t, i) for t, i in chinese if re.search(r"[a-zA-Z0-9]", t)]
    for t, tid in sorted(mixed, key=lambda x: x[1])[:20]:
        print(f"  [{tid:6d}]  {t}")
    if len(mixed) > 20:
        print(f"  ... (共 {len(mixed)} 个)")

    # ── 8. 按字符统计覆盖度 ────────────────────────────────
    print()
    print("=" * 55)
    print("中文字符覆盖统计")
    print("=" * 55)
    all_cjk_chars = set()
    for t, _ in chinese:
        for ch in t:
            if has_cjk(ch):
                all_cjk_chars.add(ch)
    print(f"  词表覆盖的独立 CJK 字符数: {len(all_cjk_chars)}")
    print(f"  常用汉字 (Unicode CJK Unified): 20992")
    print(f"  覆盖比例: {len(all_cjk_chars) / 20992 * 100:.1f}%")


if __name__ == "__main__":
    main()
