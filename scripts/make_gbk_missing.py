"""生成目标字体的 GBK 缺失字集，供 inference 使用。

用法:   python scripts/make_gbk_missing.py --font fonts/target_font.ttf
输出:   charsets/gbk_all.txt
        charsets/gbk_coverage/<字体名>/covered.txt
        charsets/gbk_coverage/<字体名>/missing.txt   ← 给 inference
"""
import argparse
from pathlib import Path

from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont


def gbk_hanzi() -> set[str]:
    """枚举所有能编码进 GBK 的汉字（CJK 基本区 + 扩展 A）。"""
    chars: set[str] = set()
    for hi in range(0x81, 0xFF):
        for lo in range(0x40, 0xFF):
            if lo == 0x7F:
                continue
            try:
                ch = bytes([hi, lo]).decode("gbk")
            except UnicodeDecodeError:
                continue
            if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
                chars.add(ch)
    return chars


def missing_or_empty(glyph_set, cmap, char: str) -> bool:
    """与项目 FontCoverageAnalyzer._is_glyph_empty_or_missing 同一套判断。"""
    name = cmap.get(ord(char))
    if name is None or name not in glyph_set:
        return True
    pen = RecordingPen()
    glyph_set[name].draw(pen)
    return len(pen.value) == 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--font", default="fonts/target_font.ttf")
    p.add_argument("--out_root", default="charsets")
    p.add_argument("--charset_name", default="gbk")
    args = p.parse_args()

    font_path = Path(args.font)
    font_name = font_path.stem
    out_root = Path(args.out_root)

    all_chars = gbk_hanzi()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"{args.charset_name}_all.txt").write_text(
        "\n".join(sorted(all_chars)) + "\n", encoding="utf-8"
    )

    font = TTFont(str(font_path))
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()

    covered, missing = set(), set()
    for ch in all_chars:
        (missing if missing_or_empty(glyph_set, cmap, ch) else covered).add(ch)

    out_dir = out_root / f"{args.charset_name}_coverage" / font_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "covered.txt").write_text("\n".join(sorted(covered)) + "\n", encoding="utf-8")
    (out_dir / "missing.txt").write_text("\n".join(sorted(missing)) + "\n", encoding="utf-8")

    total = len(all_chars)
    print(f"GBK 汉字总数: {total}")
    print(f"已覆盖: {len(covered)} ({len(covered)/total:.2%})")
    print(f"缺失:   {len(missing)} ({len(missing)/total:.2%})")
    print(f"缺失字表: {out_dir / 'missing.txt'}")


if __name__ == "__main__":
    main()