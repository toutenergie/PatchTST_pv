// =============================================================================
// docx_rendu.js — Rendu Word (docx-js) d'un document décrit en JSON.
//
// Usage : node docx_rendu.js document.json sortie.docx
//
// Le JSON contient { titre, sous_titre, auteurs, affiliation, entete, blocs[] }.
// Types de blocs : h1, h2, h3, p, puces, numeros, figure, tableau, equation,
//                  saut_page, references, encadre.
// Mise en forme en ligne dans le texte : **gras**, *italique*, `code`,
// ^exposant^ et _{indice}.
// =============================================================================
const fs = require("fs");
const path = require("path");
const d = require("docx");

const [, , entree, sortie] = process.argv;
const doc = JSON.parse(fs.readFileSync(entree, "utf8"));

const POLICE = "Times New Roman";
const TAILLE = 22;              // demi-points : 11 pt
const LARGEUR_UTILE = 9026;     // A4 (11906) - marges 2 × 1440 DXA

// --- Texte enrichi --------------------------------------------------------------
function runs(texte, base = {}) {
  const res = [];
  const motif = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\^[^^]+\^|_\{[^}]+\})/g;
  let dernier = 0, m;
  while ((m = motif.exec(texte)) !== null) {
    if (m.index > dernier) res.push(new d.TextRun({ text: texte.slice(dernier, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) res.push(new d.TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) res.push(new d.TextRun({ text: t.slice(1, -1), font: "Consolas", size: (base.size || TAILLE) - 2, ...base, }));
    else if (t.startsWith("^")) res.push(new d.TextRun({ text: t.slice(1, -1), superScript: true, ...base }));
    else if (t.startsWith("_{")) res.push(new d.TextRun({ text: t.slice(2, -1), subScript: true, ...base }));
    else res.push(new d.TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    dernier = m.index + t.length;
  }
  if (dernier < texte.length) res.push(new d.TextRun({ text: texte.slice(dernier), ...base }));
  return res;
}

const para = (texte, opts = {}) => new d.Paragraph({
  children: runs(texte, opts.run || {}),
  alignment: opts.alignement || d.AlignmentType.JUSTIFIED,
  spacing: { after: opts.apres ?? 120, line: 276 },
  indent: opts.retrait,
  style: opts.style,
  keepNext: opts.keepNext,
});

// --- Tableaux -------------------------------------------------------------------
const bord = { style: d.BorderStyle.SINGLE, size: 4, color: "7F8C8D" };
function tableau(b) {
  const n = b.entetes.length;
  const poids = b.largeurs || Array(n).fill(1);
  const total = poids.reduce((a, x) => a + x, 0);
  const larg = poids.map(x => Math.floor(LARGEUR_UTILE * x / total));
  larg[n - 1] += LARGEUR_UTILE - larg.reduce((a, x) => a + x, 0);
  const taille = b.taille || 16;
  const cellule = (texte, j, entete, ligne) => new d.TableCell({
    width: { size: larg[j], type: d.WidthType.DXA },
    borders: { top: bord, bottom: bord, left: bord, right: bord },
    shading: entete ? { fill: "D6E4F0", type: d.ShadingType.CLEAR, color: "auto" }
      : (b.surligner && b.surligner.includes(ligne)) ? { fill: "FDECEA", type: d.ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new d.Paragraph({
      children: runs(String(texte), { size: taille, bold: entete || undefined }),
      alignment: j === 0 || (b.gauche && b.gauche.includes(j)) ? d.AlignmentType.LEFT : d.AlignmentType.CENTER,
    })],
  });
  const lignes = [new d.TableRow({ tableHeader: true, children: b.entetes.map((t, j) => cellule(t, j, true, -1)) })];
  b.lignes.forEach((l, i) => lignes.push(new d.TableRow({ cantSplit: true, children: l.map((t, j) => cellule(t, j, false, i)) })));
  return new d.Table({ width: { size: LARGEUR_UTILE, type: d.WidthType.DXA }, columnWidths: larg, rows: lignes });
}

const legende = (texte) => new d.Paragraph({
  children: runs(texte, { size: 19 }), alignment: d.AlignmentType.CENTER, spacing: { before: 60, after: 200 },
});

// --- Blocs ----------------------------------------------------------------------
const enfants = [];
// Page de titre
enfants.push(new d.Paragraph({ children: runs(doc.titre, { bold: true, size: 32 }), alignment: d.AlignmentType.CENTER, spacing: { after: 200 } }));
if (doc.sous_titre) enfants.push(new d.Paragraph({ children: runs(doc.sous_titre, { italics: true, size: 24 }), alignment: d.AlignmentType.CENTER, spacing: { after: 200 } }));
if (doc.auteurs) enfants.push(new d.Paragraph({ children: runs(doc.auteurs, { size: 22 }), alignment: d.AlignmentType.CENTER, spacing: { after: 60 } }));
if (doc.affiliation) enfants.push(new d.Paragraph({ children: runs(doc.affiliation, { size: 19, italics: true }), alignment: d.AlignmentType.CENTER, spacing: { after: 300 } }));

for (const b of doc.blocs) {
  switch (b.type) {
    case "h1": enfants.push(new d.Paragraph({ heading: d.HeadingLevel.HEADING_1, children: runs(b.texte), keepNext: true })); break;
    case "h2": enfants.push(new d.Paragraph({ heading: d.HeadingLevel.HEADING_2, children: runs(b.texte), keepNext: true })); break;
    case "h3": enfants.push(new d.Paragraph({ heading: d.HeadingLevel.HEADING_3, children: runs(b.texte), keepNext: true })); break;
    case "p": enfants.push(para(b.texte)); break;
    case "encadre":
      enfants.push(new d.Paragraph({
        children: runs(b.texte, { size: 20 }), alignment: d.AlignmentType.JUSTIFIED, spacing: { before: 80, after: 160 },
        shading: { fill: "F4F6F6", type: d.ShadingType.CLEAR, color: "auto" },
        border: { left: { style: d.BorderStyle.SINGLE, size: 18, color: "C0392B", space: 6 } },
        indent: { left: 200, right: 200 },
      })); break;
    case "puces": for (const t of b.items) enfants.push(new d.Paragraph({ children: runs(t), numbering: { reference: "puces", level: 0 }, alignment: d.AlignmentType.JUSTIFIED, spacing: { after: 60 } })); break;
    case "numeros": for (const t of b.items) enfants.push(new d.Paragraph({ children: runs(t), numbering: { reference: "numeros", level: 0, instance: b.instance || 1 }, alignment: d.AlignmentType.JUSTIFIED, spacing: { after: 60 } })); break;
    case "equation":
      // tabulation centrée (équation) + tabulation droite (numéro)
      enfants.push(new d.Paragraph({
        tabStops: [{ type: d.TabStopType.CENTER, position: Math.round(LARGEUR_UTILE / 2) }, { type: d.TabStopType.RIGHT, position: LARGEUR_UTILE }],
        children: [new d.TextRun({ children: [new d.Tab()] }), ...runs(b.texte, { font: "Cambria Math" }),
                   new d.TextRun({ children: [new d.Tab(), `(${b.numero})`] })],
        alignment: d.AlignmentType.LEFT, spacing: { before: 80, after: 120 },
      })); break;
    case "figure": {
      const img = fs.readFileSync(b.chemin);
      const largeur = Math.round((b.largeur_cm || 16) / 2.54 * 96);
      const hauteur = Math.round(largeur * b.ratio);
      enfants.push(new d.Paragraph({ children: [new d.ImageRun({ type: "png", data: img, transformation: { width: largeur, height: hauteur }, altText: { title: b.legende.slice(0, 60), description: b.legende, name: path.basename(b.chemin) } })], alignment: d.AlignmentType.CENTER, keepNext: true, spacing: { before: 120 } }));
      enfants.push(legende(b.legende));
      break;
    }
    case "tableau":
      enfants.push(new d.Paragraph({ children: runs(b.legende, { size: 19 }), alignment: d.AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 } }));
      enfants.push(tableau(b));
      if (b.note) enfants.push(new d.Paragraph({ children: runs(b.note, { size: 17, italics: true }), spacing: { before: 40, after: 200 } }));
      else enfants.push(new d.Paragraph({ children: [], spacing: { after: 120 } }));
      break;
    case "saut_page": enfants.push(new d.Paragraph({ children: [new d.PageBreak()] })); break;
    case "references":
      for (const r of b.items) enfants.push(new d.Paragraph({ children: runs(r, { size: 19 }), alignment: d.AlignmentType.LEFT, indent: { left: 454, hanging: 454 }, spacing: { after: 60 } }));
      break;
    default: throw new Error("type de bloc inconnu : " + b.type);
  }
}

const document = new d.Document({
  creator: doc.auteurs || "",
  title: doc.titre,
  styles: {
    default: { document: { run: { font: POLICE, size: TAILLE } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, font: POLICE, color: "1F3A5F" }, paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, font: POLICE, color: "1F3A5F" }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, italics: true, font: POLICE }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "puces", levels: [{ level: 0, format: d.LevelFormat.BULLET, text: "•", alignment: d.AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      { reference: "numeros", levels: [{ level: 0, format: d.LevelFormat.DECIMAL, text: "%1.", alignment: d.AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: { default: new d.Header({ children: [new d.Paragraph({ children: runs(doc.entete || "", { size: 16, italics: true, color: "7F8C8D" }), alignment: d.AlignmentType.RIGHT })] }) },
    footers: { default: new d.Footer({ children: [new d.Paragraph({ alignment: d.AlignmentType.CENTER, children: [new d.TextRun({ children: [d.PageNumber.CURRENT], size: 18 })] })] }) },
    children: enfants,
  }],
});

d.Packer.toBuffer(document).then(buf => { fs.writeFileSync(sortie, buf); console.log("écrit :", sortie); });
