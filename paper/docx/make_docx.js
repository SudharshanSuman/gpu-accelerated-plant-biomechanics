/*
 * Builds an IEEE-format Word rendering of the manuscript.
 *
 * Lives in the repository rather than a scratch directory on purpose: this is
 * a build script for a deliverable, and an earlier copy was lost to a /tmp
 * wipe on reboot.
 *
 *   cd paper/docx && npm install && node make_docx.js
 *
 * Numbers here are duplicated from paper/results_macros.tex by hand, which is
 * the one place this project tolerates that. Keep them in sync when the
 * generators are re-run: the LaTeX build is authoritative.
 */
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, SectionType,
  convertInchesToTwip,
} = require('docx');

const PROJ = path.resolve(__dirname, '..', '..');
const RES = path.join(PROJ, 'results') + path.sep;
const OUT = path.join(PROJ, 'paper', 'main.docx');

const PAGE = { width: 12240, height: 15840 };
const MARGIN = { top: convertInchesToTwip(0.75), bottom: convertInchesToTwip(1.0),
                 left: convertInchesToTwip(0.625), right: convertInchesToTwip(0.625) };
const TEXT_W_IN = 8.5 - 0.625 * 2;
const GUTTER_IN = 0.2;
const COL_W_IN = (TEXT_W_IN - GUTTER_IN) / 2;
const COL_PX = Math.round(COL_W_IN * 96);
const FULL_PX = Math.round(TEXT_W_IN * 96);
const FONT = 'Times New Roman';
const BODY = 20;

const body = (text, o = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 0 },
  indent: o.noIndent ? undefined : { firstLine: convertInchesToTwip(0.2) },
  children: [new TextRun({ text, font: FONT, size: BODY })],
});
const rich = (runs, o = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 0 },
  indent: o.noIndent ? undefined : { firstLine: convertInchesToTwip(0.2) },
  children: runs.map(([t, f = {}]) => new TextRun({ text: t, font: FONT, size: BODY, bold: f.bold, italics: f.italics })),
});
const ROMAN = ['0','I','II','III','IV','V','VI','VII','VIII','IX','X'];
let secN = 0, subN = 0;
const section = (t) => { secN++; subN = 0; return new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: `${ROMAN[secN]}.  ${t.toUpperCase()}`, font: FONT, size: BODY, smallCaps: true })] }); };
const sectionStar = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: t.toUpperCase(), font: FONT, size: BODY, smallCaps: true })] });
const subsection = (t) => { subN++; return new Paragraph({
  spacing: { before: 160, after: 80 },
  children: [new TextRun({ text: `${String.fromCharCode(64 + subN)}.  ${t}`, font: FONT, size: BODY, italics: true })] }); };
const equation = (t, n) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 },
  children: [new TextRun({ text: `${t}          (${n})`, font: FONT, size: BODY, italics: true })] });
const figure = (file, px, aspect, caption) => ([
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 180, after: 60 },
    children: [new ImageRun({ type: 'png', data: fs.readFileSync(RES + file),
      transformation: { width: px, height: Math.round(px * aspect) } })] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 180 },
    children: [new TextRun({ text: caption, font: FONT, size: 16 })] }),
]);
const tableCaption = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 180, after: 80 },
  children: [new TextRun({ text: t, font: FONT, size: 16, smallCaps: true })] });
const makeTable = (headers, rows) => {
  const n = headers.length, total = Math.round(COL_W_IN * 1440);
  const cw = Math.floor(total / n), widths = Array(n).fill(cw);
  widths[n - 1] = total - cw * (n - 1);
  const cell = (text, { bold = false, top = false, bottom = false, idx = 0 } = {}) => new TableCell({
    width: { size: widths[idx], type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 40, right: 40 },
    borders: {
      top: top ? { style: BorderStyle.SINGLE, size: 6, color: '000000' } : { style: BorderStyle.NONE },
      bottom: bottom ? { style: BorderStyle.SINGLE, size: 6, color: '000000' } : { style: BorderStyle.NONE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } },
    children: [new Paragraph({ alignment: idx === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
      children: [new TextRun({ text, font: FONT, size: 14, bold })] })] });
  return new Table({ columnWidths: widths, width: { size: total, type: WidthType.DXA }, rows: [
    new TableRow({ children: headers.map((h, i) => cell(h, { bold: true, top: true, bottom: true, idx: i })) }),
    ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, { bottom: ri === rows.length - 1, idx: i })) })) ] });
};

// ---------------- front matter ----------------
const half = Math.round(TEXT_W_IN * 1440 / 2);
const authorCell = (name, email) => new TableCell({
  width: { size: half, type: WidthType.DXA },
  borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
             left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } },
  children: [
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: name, font: FONT, size: 22 })] }),
    ...['Division of Computer Science and Engineering', 'Karunya Institute of Technology and Sciences', 'Coimbatore, India', email]
      .map(t => new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: t, font: FONT, size: 20, italics: true })] })),
  ] });

const front = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: 'GPU-Accelerated Plant Biomechanics Simulation for Agricultural Robots', font: FONT, size: 48 })] }),
  new Table({ columnWidths: [half, half], width: { size: half * 2, type: WidthType.DXA },
    rows: [new TableRow({ children: [
      authorCell('Sudharshan S', 'sudharshans@karunya.edu.in'),
      authorCell('S. E. Vinodh Edwards', 'ewards@karunya.edu') ] })] }),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: '' })] }),
];

module.exports = { Document, Packer, Paragraph, TextRun, AlignmentType, SectionType,
  Table, TableRow, TableCell, WidthType, BorderStyle, PAGE, MARGIN, COL_PX, FULL_PX,
  FONT, BODY, GUTTER_IN, convertInchesToTwip, body, rich, section, sectionStar,
  subsection, equation, figure, tableCaption, makeTable, front, OUT };
