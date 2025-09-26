# -*- coding: utf-8 -*-
"""
Crea CognitivaAI_Deck_Custom.pptx en Windows
- Inserta logo, QR y un slice MRI (si existen).
- Inserta imágenes de p27_final (ROC/PR/Calibración/Coste) si están presentes.
- Construye una Tabla Ejecutiva desde p25_master_table.csv si existe (mapeo de columnas robusto).
Autor: Fran Ramírez (Bootcamp Ironhack Data-PT · Marzo 2025)
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pathlib import Path
import pandas as pd

# ==========================
# Rutas LOCALES (ajústalas si quieres)
# ==========================
P27_DIR = Path(r"C:\Users\PC\Desktop\cognitiva-ai\p27_final")
P25_DIR = Path(r"C:\Users\PC\Desktop\cognitiva-ai\p25_informe_final")
LOGO    = Path(r"C:\Users\PC\Desktop\cognitiva-ai\graficos\logo3.png")
QR      = Path(r"C:\Users\PC\Desktop\cognitiva-ai\graficos\QRFran.png")
MRI     = Path(r"C:\Users\PC\Desktop\cognitiva-ai\data\OAS2_PROCESSED\OAS2_0186_MR3_slice15.png")
OUT     = Path(r"C:\Users\PC\Desktop\cognitiva-ai\CognitivaAI_Deck_Custom.pptx")

# Nombres típicos de figuras (se intentarán cargar si existen)
FIGS_WISHLIST = [
    "p25_roc_P24_TEST_OAS1.png", "p25_roc_P24_TEST_OAS2.png",
    "p25_pr_P24_TEST_OAS1.png",  "p25_pr_P24_TEST_OAS2.png",
    "p25_cal_P24_TEST_OAS1.png", "p25_cal_P24_TEST_OAS2.png",
    "p25_cost_curve_OAS1.png",   "p25_cost_curve_OAS2.png",
]
MASTER_TABLE_CSV = P25_DIR / "p25_master_table.csv"

# ==========================
# Estilo / paleta
# ==========================
BG      = RGBColor(13,17,23)
CYAN    = RGBColor(84,190,255)
AMBER   = RGBColor(255,183,77)
GREEN   = RGBColor(134,236,161)
WHITE   = RGBColor(255,255,255)
MUTED   = RGBColor(180,190,200)
TILE_BG = RGBColor(30,36,48)

prs = Presentation()
W, H = prs.slide_width, prs.slide_height

def bg(slide, color=BG):
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    r.fill.solid(); r.fill.fore_color.rgb = color
    r.line.fill.background()
    slide.shapes._spTree.remove(r._element); slide.shapes._spTree.insert(2, r._element)

def band(slide, color, y_in, h_in):
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(y_in), W, Inches(h_in))
    rect.fill.solid(); rect.fill.fore_color.rgb = color; rect.line.fill.background()
    return rect

def add_img(slide, pth, x, y, w=None, h=None):
    pth = Path(pth)
    if not pth.exists():
        return None
    return slide.shapes.add_picture(str(pth), Inches(x), Inches(y),
                                    width=Inches(w) if w else None,
                                    height=Inches(h) if h else None)

def title(slide, text, subtitle=None):
    box = slide.shapes.add_textbox(Inches(1.2), Inches(1.4), Inches(10.5), Inches(1.8))
    tf = box.text_frame
    p = tf.paragraphs[0]; r = p.add_run()
    r.text = text; r.font.size = Pt(46); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = "Montserrat"
    if subtitle:
        s = tf.add_paragraph(); s.text = subtitle; s.font.size = Pt(18); s.font.color.rgb = MUTED; s.font.name = "Inter"

def header(slide, text, color=CYAN):
    box = slide.shapes.add_textbox(Inches(0.9), Inches(0.9), Inches(10.8), Inches(0.9))
    tf = box.text_frame
    p = tf.paragraphs[0]; r = p.add_run()
    r.text = text; r.font.size = Pt(34); r.font.bold = True; r.font.color.rgb = color; r.font.name = "Montserrat"

def bullets(slide, items, x=0.9, y=1.8, w=10.5, h=5.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear()
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text = t; p.font.size = Pt(20); p.font.color.rgb = WHITE; p.font.name = "Inter"

def notes(slide, text):
    n = slide.notes_slide; tf = n.notes_text_frame; tf.clear(); tf.text = text

def kpi_tiles(slide, kpis):
    cols = len(kpis); gap = Inches(0.3)
    tile_w = (W - Inches(1.8) - gap*(cols-1)) / cols
    x = Inches(0.9); y = Inches(2.1)
    for k in kpis:
        rect = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, tile_w, Inches(1.5))
        rect.fill.solid(); rect.fill.fore_color.rgb = TILE_BG; rect.line.fill.background()
        t1 = slide.shapes.add_textbox(x+Inches(0.3), y+Inches(0.25), tile_w-Inches(0.6), Inches(0.6))
        p = t1.text_frame.paragraphs[0].add_run()
        p.text = k["value"]; p.font.size = Pt(34); p.font.bold = True
        p.font.color.rgb = k.get("color", CYAN); p.font.name = "Montserrat"
        t2 = slide.shapes.add_textbox(x+Inches(0.3), y+Inches(0.9), tile_w-Inches(0.6), Inches(0.5))
        t2f = t2.text_frame.paragraphs[0]
        t2f.text = k["label"]; t2f.font.size = Pt(16); t2f.font.color.rgb = MUTED; t2f.font.name = "Inter"
        x += tile_w + gap

# ---------- NUEVO: función robusta para la tabla ejecutiva ----------
def _pick_col(df, *cands, default=None):
    for c in cands:
        if c in df.columns:
            return c
    return default

def table_exec_from_master(slide, csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        bullets(slide, [f"No se pudo leer {csv_path}", str(e)], y=2.2)
        return

    # Mapear nombres posibles
    col_pipeline = _pick_col(df, "Pipeline", "pipeline")
    col_cohort   = _pick_col(df, "Cohort", "Cohorte", "cohort", "cohorte")
    col_method   = _pick_col(df, "Method", "Método", "Notas", "method", "método", "notas")
    col_auc      = _pick_col(df, "AUC", "auc")
    col_prauc    = _pick_col(df, "PRAUC", "PR-AUC", "PR_AUC", "prauc")
    col_brier    = _pick_col(df, "Brier", "BRIER", "BrierScore", "brier")

    # Validación mínima
    missing = [n for n,v in {
        "Pipeline":col_pipeline, "Cohort/Cohorte":col_cohort,
        "AUC":col_auc, "PR-AUC":col_prauc, "Brier":col_brier
    }.items() if v is None]
    if missing:
        bullets(slide, [f"Faltan columnas en master_table: {', '.join(missing)}"], y=2.2)
        return

    # Si no hay 'Method/Método/Notas', inventamos columna vacía
    if col_method is None:
        df["_MetodoTmp"] = "—"
        col_method = "_MetodoTmp"

    # Filtrar filas de interés (P24/P26 y ALL/OAS1/OAS2 si existen)
    mask_pipe = df[col_pipeline].isin(["P24","P26"])
    if col_cohort:
        mask_coh = df[col_cohort].isin(["ALL","OAS1","OAS2"])
        keep_df = df[mask_pipe & mask_coh].copy()
    else:
        keep_df = df[mask_pipe].copy()

    # Columnas normalizadas para pintar
    cols_norm = {
        "Pipeline": col_pipeline,
        "Cohorte":  col_cohort if col_cohort else col_cohort,  # puede ser None, no pasa nada
        "Método":   col_method,
        "AUC":      col_auc,
        "PR-AUC":   col_prauc,
        "Brier":    col_brier
    }
    # Construimos dataframe para tabla (sólo columnas presentes)
    present_cols = {k:v for k,v in cols_norm.items() if v is not None}
    table_df = keep_df[list(present_cols.values())].copy()
    table_df.columns = list(present_cols.keys())

    # Si Cohorte no existe, creamos con '—'
    if "Cohorte" not in table_df.columns:
        table_df["Cohorte"] = "—"

    rows = table_df.to_dict("records")
    if not rows:
        bullets(slide, ["La tabla ejecutiva está vacía o no coincide el formato."], y=2.2)
        return

    # Construir tabla PPT
    cols_display = ["Pipeline","Cohorte","Método","AUC","PR-AUC","Brier"]
    nrows = len(rows) + 1
    tbl_shape = slide.shapes.add_table(nrows, len(cols_display), Inches(0.6), Inches(1.6), Inches(12.8), Inches(4.6))
    tbl = tbl_shape.table
    # header
    for j,c in enumerate(cols_display):
        cell = tbl.cell(0,j); cell.text = c
        run = cell.text_frame.paragraphs[0].runs[0]; run.font.bold = True
    # rows
    for i, r in enumerate(rows, start=1):
        for j, c in enumerate(cols_display):
            v = "" if pd.isna(r.get(c,"")) else str(r.get(c,""))
            tbl.cell(i,j).text = v

def grid_images(slide, paths, cols=2, cell_w=6.0, cell_h=3.4, x0=0.6, y0=1.8, xgap=0.4, ygap=0.4):
    r, c = 0, 0
    for p in paths:
        p = Path(p)
        if not p.exists(): 
            continue
        x = x0 + c*(cell_w + xgap)
        y = y0 + r*(cell_h + ygap)
        add_img(slide, p, x, y, w=cell_w)
        c += 1
        if c >= cols:
            c = 0; r += 1

# ========= Slide 1: Cover =========
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
band(s, CYAN, y_in=0.65, h_in=0.15); band(s, AMBER, y_in=5.2, h_in=0.12)
add_img(s, LOGO, x=0.8, y=0.4, w=2.0)
title(s, "Cognitiva-AI",
      "Predicción temprana de demencia con datos clínicos + MRI (P27 • Política S2)\nAutor: Fran Ramírez · Bootcamp Ironhack Data-PT · Marzo 2025")

# ========= Slide 2: Problema y objetivo =========
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Problema y objetivo")
bullets(s, [
    "Riesgo binario: Nondemented vs Demented/Converted",
    "Probabilidades calibradas + umbrales por cohorte",
    "Criterio clínico: priorizar recall (minimizar FN)"])
add_img(s, MRI, x=9.2, y=1.2, w=3.2)

# ========= Slide 3: Datos y preprocesado =========
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Datos y preprocesado (OASIS-1 / OASIS-2)")
bullets(s, [
    "MRI: 20 slices axiales, z-score + CLAHE; pooling por paciente",
    "Clínico: MMSE, CDR, eTIV, nWBV… imputación y one-hot sexo",
    "OASIS-2: 1ª visita por paciente; splits estratificados por paciente"])

# ========= Slide 4: Cronología y arquitectura =========
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Cronología y arquitectura (P1→P27)")
bullets(s, [
    "P24: Meta simple (LR elastic-net + Platt) — imagen",
    "P26: Intermodal con fusión Late (mejor que Mid)",
    "P26b: Platt por cohorte → mejora Brier en OAS1",
    "P27: Política S2 + release reproducible"])

# ========= Slide 5: Resultados (KPIs) =========
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Resultados (TEST)")
bullets(s, [
    "P24 (imagen): AUC ALL 0.727 · OAS1 0.754 · OAS2 0.750",
    "P26 Late (raw): AUC ALL 0.713",
    "P26b: OAS1 AUC ~0.754 · OAS2 ~0.652",
    "Política S2: OAS2 Recall ≈ 0.92 (thr≈0.493)"])
# KPIs en tarjetas
def kpi_tiles(slide, kpis):
    cols = len(kpis); gap = Inches(0.3)
    tile_w = (W - Inches(1.8) - gap*(cols-1)) / cols
    x = Inches(0.9); y = Inches(2.1)
    for k in kpis:
        rect = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, tile_w, Inches(1.5))
        rect.fill.solid(); rect.fill.fore_color.rgb = TILE_BG; rect.line.fill.background()
        t1 = slide.shapes.add_textbox(x+Inches(0.3), y+Inches(0.25), tile_w-Inches(0.6), Inches(0.6))
        p = t1.text_frame.paragraphs[0].add_run()
        p.text = k["value"]; p.font.size = Pt(34); p.font.bold = True
        p.font.color.rgb = k.get("color", CYAN); p.font.name = "Montserrat"
        t2 = slide.shapes.add_textbox(x+Inches(0.3), y+Inches(0.9), tile_w-Inches(0.6), Inches(0.5))
        t2f = t2.text_frame.paragraphs[0]
        t2f.text = k["label"]; t2f.font.size = Pt(16); t2f.font.color.rgb = MUTED; t2f.font.name = "Inter"
        x += tile_w + gap

kpi_tiles(s, [
    {"label":"AUC P24 (OAS1)","value":"0.754","color":CYAN},
    {"label":"AUC P24 (OAS2)","value":"0.750","color":GREEN},
    {"label":"Recall S2 (OAS2)","value":"0.92","color":AMBER},
])

# ========= Slide 6: Política S2 =========
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Política S2 — decisión por coste")
bullets(s, [
    "Base 5:1 (FN:FP) por cohorte",
    "Ajuste OAS2 para Recall ≥ 0.90 (cribado)",
    "Umbrales activos: OAS1 = 0.42 · OAS2 ≈ 0.493"])

# ========= Slide 7: Tabla ejecutiva (desde CSV) =========
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Tabla ejecutiva (TEST)")
if MASTER_TABLE_CSV.exists():
    table_exec_from_master(s, MASTER_TABLE_CSV)
else:
    bullets(s, [f"No se encontró {MASTER_TABLE_CSV}. Añádelo para la tabla."], y=2.2)

# ========= Slide 8: Gráficos de p27_final =========
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Gráficos (ROC/PR/Calibración/Coste)")
# construye lista con los que existan
paths = [P27_DIR / n for n in FIGS_WISHLIST if (P27_DIR / n).exists()]
# si no hay coincidencias, inserta cualquier PNG de la carpeta (hasta 4)
if not paths:
    paths = list(P27_DIR.glob("*.png"))[:4]
def grid_images(slide, paths, cols=2, cell_w=6.1, cell_h=3.4, x0=0.7, y0=1.6, xgap=0.4, ygap=0.4):
    r, c = 0, 0
    for p in paths:
        p = Path(p)
        if not p.exists(): 
            continue
        x = x0 + c*(cell_w + xgap)
        y = y0 + r*(cell_h + ygap)
        add_img(slide, p, x, y, w=cell_w)
        c += 1
        if c >= cols:
            c = 0; r += 1
if paths:
    grid_images(s, paths)
else:
    bullets(s, [f"No se encontraron PNG en {P27_DIR}. Añade tus figuras y re-ejecuta."], y=2.2)

# ========= Slide 9: Demo y release =========
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Demo y release")
bullets(s, [
    "Streamlit: Modo Demo + sliders de umbral/política",
    "FastAPI: /predict (clínica + p_img)",
    "Zip: p26_release.zip (modelos, config S2, QA, docs)"])

# ========= Slide 10: Limitaciones y próximos pasos =========
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Limitaciones y próximos pasos")
bullets(s, [
    "Validación externa (ADNI) · aumentar N",
    "Recalibración periódica si ECE > 0.2",
    "Explorar 2.5D/3D y explicabilidad (SHAP/Grad-CAM)"])

# ========= Slide 11: Cierre con QR =========
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
title(s, "Gracias", "Conecta en LinkedIn")
add_img(s, QR, x=8.0, y=2.4, w=3.2)
# mini badge “in”
badge = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.7), Inches(2.3), Inches(0.9), Inches(0.9))
badge.fill.solid(); badge.fill.fore_color.rgb = CYAN; badge.line.fill.background()
tb = s.shapes.add_textbox(Inches(7.85), Inches(2.42), Inches(0.7), Inches(0.7))
tbf = tb.text_frame.paragraphs[0]; tbf.text = "in"
tbf.runs[0].font.size = Pt(40); tbf.runs[0].font.bold = True; tbf.runs[0].font.color.rgb = WHITE

# Guardar
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"✅ Deck creado: {OUT}")

