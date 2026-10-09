// =============================================================================
// docx_render.js — Word rendering (docx-js) of a document described in JSON.
//
// Usage: node docx_render.js document.json output.docx
//
// The JSON holds { title, subtitle, authors, affiliation, header, blocks[] }.
// Block types: h1, h2, h3, p, bullets, numbered, figure, table, equation,
//              page_break, references, box.
// Inline formatting in text: **bold**, *italic*, `code`, ^superscript^ and
// _{subscript}.
// =============================================================================
const fs = require("fs");
const path = require("path");
const d = require("docx");

const [, , input, output] = process.argv;
const doc = JSON.parse(fs.readFileSync(input, "utf8"));

const FONT = "Times New Roman";
const SIZE = 22;                // half-points: 11 pt
const USABLE_WIDTH = 9026;      // A4 (11906) - margins 2 × 1440 DXA

// --- Rich text ------------------------------------------------------------------
function runs(text, base = {}) {
  const res = [];
  const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\^[^^]+\^|_\{[^}]+\})/g;
  let last = 0, m;
  while ((m = pattern.exec(text)) !== null) {
    if (m.index > last) res.push(new d.TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) res.push(new d.TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) res.push(new d.TextRun({ text: t.slice(1, -1), font: "Consolas", size: (base.size || SIZE) - 2, ...base }));
    else if (t.startsWith("^")) res.push(new d.TextRun({ text: t.slice(1, -1), superScript: true, ...base }));
    else if (t.startsWith("_{")) res.push(new d.TextRun({ text: t.slice(2, -1), subScript: true, ...base }));
    else res.push(new d.TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) res.push(new d.TextRun({ text: text.slice(last), ...base }));
  return res;
}

const paragraph = (text) => new d.Paragraph({
  children: runs(text), alignment: d.AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 },
});

// --- Tables ---------------------------------------------------------------------
const border = { style: d.BorderStyle.SINGLE, size: 4, color: "7F8C8D" };
function table(b) {
  const n = b.headers.length;
  const weights = b.widths || Array(n).fill(1);
  const total = weights.reduce((a, x) => a + x, 0);
  const widths = weights.map(x => Math.floor(USABLE_WIDTH * x / total));
  widths[n - 1] += USABLE_WIDTH - widths.reduce((a, x) => a + x, 0);
  const size = b.size || 16;
  const cell = (text, j, header, row) => new d.TableCell({
    width: { size: widths[j], type: d.WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: header ? { fill: "D6E4F0", type: d.ShadingType.CLEAR, color: "auto" }
      : (b.highlight && b.highlight.includes(row)) ? { fill: "FDECEA", type: d.ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new d.Paragraph({
      children: runs(String(text), { size, bold: header || undefined }),
      alignment: j === 0 || (b.left && b.left.includes(j)) ? d.AlignmentType.LEFT : d.AlignmentType.CENTER,
    })],
  });
  const rows = [new d.TableRow({ tableHeader: true, children: b.headers.map((t, j) => cell(t, j, true, -1)) })];
  b.rows.forEach((r, i) => rows.push(new d.TableRow({ cantSplit: true, children: r.map((t, j) => cell(t, j, false, i)) })));
  return new d.Table({ width: { size: USABLE_WIDTH, type: d.WidthType.DXA }, columnWidths: widths, rows });
}

const caption = (text) => new d.Paragraph({
  children: runs(text, { size: 19 }), alignment: d.AlignmentType.CENTER, spacing: { before: 60, after: 200 },
});

// --- Blocks ---------------------------------------------------------------------
const children = [];
// Title page
children.push(new d.Paragraph({ children: runs(doc.title, { bold: true, size: 32 }), alignment: d.AlignmentType.CENTER, spacing: { after: 200 } }));
if (doc.subtitle) children.push(new d.Paragraph({ children: runs(doc.subtitle, { italics: true, size: 24 }), alignment: d.AlignmentType.CENTER, spacing: { after: 200 } }));
if (doc.authors) children.push(new d.Paragraph({ children: runs(doc.authors, { size: 22 }), alignment: d.AlignmentType.CENTER, spacing: { after: 60 } }));
if (doc.affiliation) children.push(new d.Paragraph({ children: runs(doc.affiliation, { size: 19, italics: true }), alignment: d.AlignmentType.CENTER, spacing: { after: 300 } }));

for (const b of doc.blocks) {
  switch (b.type) {
    case "h1": children.push(new d.Paragraph({ heading: d.HeadingLevel.HEADING_1, children: runs(b.text), keepNext: true })); break;
    case "h2": children.push(new d.Paragraph({ heading: d.HeadingLevel.HEADING_2, children: runs(b.text), keepNext: true })); break;
    case "h3": children.push(new d.Paragraph({ heading: d.HeadingLevel.HEADING_3, children: runs(b.text), keepNext: true })); break;
    case "p": children.push(paragraph(b.text)); break;
    case "box":
      children.push(new d.Paragraph({
        children: runs(b.text, { size: 20 }), alignment: d.AlignmentType.JUSTIFIED, spacing: { before: 80, after: 160 },
        shading: { fill: "F4F6F6", type: d.ShadingType.CLEAR, color: "auto" },
        border: { left: { style: d.BorderStyle.SINGLE, size: 18, color: "C0392B", space: 6 } },
        indent: { left: 200, right: 200 },
      })); break;
    case "bullets": for (const t of b.items) children.push(new d.Paragraph({ children: runs(t), numbering: { reference: "bullets", level: 0 }, alignment: d.AlignmentType.JUSTIFIED, spacing: { after: 60 } })); break;
    case "numbered": for (const t of b.items) children.push(new d.Paragraph({ children: runs(t), numbering: { reference: "numbers", level: 0, instance: b.instance || 1 }, alignment: d.AlignmentType.JUSTIFIED, spacing: { after: 60 } })); break;
    case "equation":
      // centred tab (equation) + right tab (number)
      children.push(new d.Paragraph({
        tabStops: [{ type: d.TabStopType.CENTER, position: Math.round(USABLE_WIDTH / 2) }, { type: d.TabStopType.RIGHT, position: USABLE_WIDTH }],
        children: [new d.TextRun({ children: [new d.Tab()] }), ...runs(b.text, { font: "Cambria Math" }),
                   new d.TextRun({ children: [new d.Tab(), `(${b.number})`] })],
        alignment: d.AlignmentType.LEFT, spacing: { before: 80, after: 120 },
      })); break;
    case "figure": {
      const img = fs.readFileSync(b.path);
      const width = Math.round((b.width_cm || 16) / 2.54 * 96);
      const height = Math.round(width * b.ratio);
      children.push(new d.Paragraph({ children: [new d.ImageRun({ type: "png", data: img, transformation: { width, height }, altText: { title: b.caption.slice(0, 60), description: b.caption, name: path.basename(b.path) } })], alignment: d.AlignmentType.CENTER, keepNext: true, spacing: { before: 120 } }));
      children.push(caption(b.caption));
      break;
    }
    case "table":
      children.push(new d.Paragraph({ children: runs(b.caption, { size: 19 }), alignment: d.AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 } }));
      children.push(table(b));
      if (b.note) children.push(new d.Paragraph({ children: runs(b.note, { size: 17, italics: true }), spacing: { before: 40, after: 200 } }));
      else children.push(new d.Paragraph({ children: [], spacing: { after: 120 } }));
      break;
    case "page_break": children.push(new d.Paragraph({ children: [new d.PageBreak()] })); break;
    case "references":
      for (const r of b.items) children.push(new d.Paragraph({ children: runs(r, { size: 19 }), alignment: d.AlignmentType.LEFT, indent: { left: 454, hanging: 454 }, spacing: { after: 60 } }));
      break;
    default: throw new Error("unknown block type: " + b.type);
  }
}

const document = new d.Document({
  creator: doc.authors || "",
  title: doc.title,
  styles: {
    default: { document: { run: { font: FONT, size: SIZE } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, font: FONT, color: "1F3A5F" }, paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, font: FONT, color: "1F3A5F" }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: d.LevelFormat.BULLET, text: "•", alignment: d.AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: d.LevelFormat.DECIMAL, text: "%1.", alignment: d.AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: { default: new d.Header({ children: [new d.Paragraph({ children: runs(doc.header || "", { size: 16, italics: true, color: "7F8C8D" }), alignment: d.AlignmentType.RIGHT })] }) },
    footers: { default: new d.Footer({ children: [new d.Paragraph({ alignment: d.AlignmentType.CENTER, children: [new d.TextRun({ children: [d.PageNumber.CURRENT], size: 18 })] })] }) },
    children,
  }],
});

d.Packer.toBuffer(document).then(buf => { fs.writeFileSync(output, buf); console.log("written:", output); });
