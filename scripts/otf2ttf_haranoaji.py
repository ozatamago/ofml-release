"""原ゴシック（Harano Aji Gothic、CFF の OpenType）を、glyf の TrueType に変換する（2026-11-10）。

matplotlib の PDF 出力（pdf.fonttype 42）は、CFF の OTF を TrueType として埋め込むので、PDF の閲覧ソフト
（mupdf、ghostscript で確認）で文字が出ない。Type 3（pdf.fonttype 3）では、日本語と数式が同じラベルに入ると
保存に失敗する。そこで、輪郭を 2 次ベジェに変換した TrueType を作り、日本語版の図ではそれを使う。

usage: otf2ttf_haranoaji.py <原ゴシックの OTF のディレクトリ>
（出力：OFML_JA_FONT_DIR、既定はリポジトリの fonts/ の HaranoAjiGothic-{Medium,Bold}-glyf.ttf）
原ゴシックは https://github.com/trueroad/HaranoAjiFonts から入手できる（SIL Open Font License 1.1）。
"""
import os
import sys

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ofml_paths  # noqa: E402
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/.TinyTeX/texmf-dist/fonts/opentype/public/haranoaji')
OUT = str(ofml_paths.JA_FONT_DIR)
MAX_ERR = 1.0   # font units (1000 per em): the quadratic approximation error


def convert(src, dst):
    f = TTFont(src)
    order = f.getGlyphOrder()
    gs = f.getGlyphSet()
    glyf = newTable('glyf')
    glyf.glyphOrder = order
    glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, MAX_ERR, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    for tag in ('CFF ', 'VORG'):
        if tag in f:
            del f[tag]
    f['glyf'] = glyf
    f['loca'] = newTable('loca')
    f['head'].glyphDataFormat = 0
    maxp = f['maxp']
    maxp.tableVersion = 0x00010000
    for k in ('maxZones', 'maxTwilightPoints', 'maxStorage', 'maxFunctionDefs', 'maxInstructionDefs',
              'maxStackElements', 'maxSizeOfInstructions', 'maxComponentElements'):
        setattr(maxp, k, 0)
    maxp.maxZones = 1
    maxp.maxComponentDepth = 0
    f['post'].formatType = 3.0          # no glyph names (CID-keyed source)
    f.sfntVersion = '\x00\x01\x00\x00'
    f.save(dst)


def main():
    os.makedirs(OUT, exist_ok=True)
    for w in ('Medium', 'Bold'):
        dst = os.path.join(OUT, f'HaranoAjiGothic-{w}-glyf.ttf')
        convert(os.path.join(SRC, f'HaranoAjiGothic-{w}.otf'), dst)
        print('wrote', dst, os.path.getsize(dst))


if __name__ == '__main__':
    sys.exit(main())
