from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

# ======= Paleta / “marca” =======
BRAND_BG = RGBColor(13, 17, 23)          # fondo oscuro elegante
C_PRIMARY = RGBColor(84, 190, 255)       # cyan
C_ACCENT  = RGBColor(255, 183, 77)       # ámbar
C_OK      = RGBColor(134, 236, 161)      # verde suave
WHITE     = RGBColor(255, 255, 255)
MUTED     = RGBColor(180, 190, 200)

prs = Presentation()
W, H = prs.slide_width, prs.slide_height

def bg(slide, color=BRAND_BG):
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    r.fill.solid()
    r.fill.fore_color.rgb = color
    r.line.fill.background()
    # enviar al fondo
    slide.shapes._spTree.remove(r._element)
    slide.shapes._spTree.insert(2, r._element)

def band(slide, color, y_in=0.7, h_in=0.18, xpad=0.8, wpad=0.8):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(xpad), Inches(y_in),
        W - Inches(xpad + wpad), Inches(h_in)
    )
    shape.fill.solid(); shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def title(slide, text, subtitle=None):
    box = slide.shapes.add_textbox(Inches(0.8), Inches(1.2), Inches(11), Inches(2))
    tf = box.text_frame
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(48); r.font.bold = True; r.font.color.rgb = WHITE
    r.font.name = "Montserrat"   # si no está instalada, PowerPoint sustituirá
    if subtitle:
        p2 = tf.add_paragraph(); p2.text = subtitle
        p2.font.size = Pt(20); p2.font.color.rgb = MUTED

def header(slide, text, x=0.8, y=1.0, color=C_PRIMARY):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(10.2), Inches(0.8))
    tf = box.text_frame; p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(34); r.font.bold = True; r.font.color.rgb = color
    r.font.name = "Montserrat"

def bullets(slide, items, x=0.8, y=2.0, w=10.2, h=4.6):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear()
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t
        p.level = 0
        p.font.size = Pt(20); p.font.color.rgb = WHITE; p.font.name = "Inter"

def notes(slide, text):
    ns = slide.notes_slide
    tf = ns.notes_text_frame
    tf.clear(); tf.text = text

def kpi_tiles(slide, kpis):  # kpis=[{"label":"..","value":"..","color":RGBColor(...)}]
    cols = len(kpis)
    gap = Inches(0.3)
    tile_w = (W - Inches(1.6) - gap*(cols-1)) / cols
    x = Inches(0.8); y = Inches(2.2)
    for k in kpis:
        rect = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, tile_w, Inches(1.6))
        rect.fill.solid(); rect.fill.fore_color.rgb = RGBColor(30,36,48)
        rect.line.fill.background()
        # value
        vbox = slide.shapes.add_textbox(x+Inches(0.3), y+Inches(0.25), tile_w-Inches(0.6), Inches(0.6))
        v = vbox.text_frame.paragraphs[0].add_run()
        v.text = k["value"]; v.font.size = Pt(36); v.font.bold = True
        v.font.color.rgb = k.get("color", C_PRIMARY); v.font.name = "Montserrat"
        # label
        lbox = slide.shapes.add_textbox(x+Inches(0.3), y+Inches(0.95), tile_w-Inches(0.6), Inches(0.5))
        l = lbox.text_frame.paragraphs[0]
        l.text = k["label"]; l.font.size = Pt(16); l.font.color.rgb = MUTED; l.font.name = "Inter"
        x = x + tile_w + gap

# ======= Slide 1 — Cover =======
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
band(s, C_PRIMARY, y_in=0.65, h_in=0.20); band(s, C_ACCENT, y_in=5.1, h_in=0.12)
title(s, "Cognitiva-AI", "Predicción temprana de demencia con datos clínicos + MRI (P27 • Política S2)")
notes(s, "Apertura (30–45s): valor de combinar clínico + imagen; foco en recall y calibración.")

# ======= Slide 2 — Problema / Objetivo =======
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Problema y objetivo")
bullets(s, [
    "Riesgo binario: Nondemented vs Demented/Converted",
    "Probabilidades calibradas + umbrales por cohorte",
    "Priorizar recall (minimizar FN en cribado clínico)",
])
notes(s, "Motivar coste de FN frente a FP: intervención temprana, derivación y seguimiento.")

# ======= Slide 3 — Datos / Preprocesado =======
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Datos y preprocesado")
bullets(s, [
    "OASIS-1 (transversal) y OASIS-2 (longitudinal, 1ª visita por sujeto)",
    "MRI: 20 slices, z-score + CLAHE; pooling por paciente",
    "Clínico: MMSE, CDR, eTIV, nWBV… imputación y one-hot sexo",
    "Split por paciente; sin fuga temporal",
])
notes(s, "Aclarar elección 1ª visita en OASIS-2 para evitar leakage.")

# ======= Slide 4 — Cronología / Arquitectura =======
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Cronología (P1→P27) y arquitectura")
bullets(s, [
    "P24: Meta simple (LR elastic-net + Platt) — imagen",
    "P26: Fusión intermodal (Late > Mid)",
    "P26b: Platt por cohorte (mejora Brier en OAS1)",
    "P27: Política S2 + release reproducible",
])
notes(s, "Enmarcar el viaje: baselines → intermodal → despliegue con política.")

# ======= Slide 5 — Resultados (KPIs) =======
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Resultados (TEST)")
bullets(s, [
    "P24: AUC ALL 0.727 | OAS1 0.754 | OAS2 0.750",
    "P26 Late (raw): AUC ALL 0.713",
    "P26b: OAS1 AUC ~0.754 | OAS2 ~0.652",
    "S2: OAS2 recall ≈0.92 (thr≈0.493)",
])
kpi_tiles(s, [
    {"label": "AUC P24 (OAS1)", "value": "0.754", "color": C_PRIMARY},
    {"label": "AUC P24 (OAS2)", "value": "0.750", "color": C_OK},
    {"label": "Recall S2 (OAS2)", "value": "0.92",  "color": C_ACCENT},
])
notes(s, "Trade-offs: aumentar sensibilidad en OAS2 manteniendo coste razonable.")

# ======= Slide 6 — Política S2 =======
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Política S2 — decisión por coste")
bullets(s, [
    "Base 5:1 (FN:FP) por cohorte",
    "Ajuste OAS2 para Recall ≥ 0.90 (cribado)",
    "Umbrales activos: OAS1=0.42 · OAS2≈0.493",
])
notes(s, "Justificar política explícita y auditable (deployment_config.json).")

# ======= Slide 7 — Demo y Release =======
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Demo y release")
bullets(s, [
    "Streamlit: Modo Demo + sliders de umbral y política",
    "FastAPI: /predict (clínica + p_img)",
    "Zip: p26_release.zip (modelos, config S2, QA, docs)",
])
notes(s, "Proponer demo de 60–90s: cargar CSV demo, mover umbrales, ver coste/metrics.")

# ======= Slide 8 — Limitaciones / Próximos pasos =======
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
header(s, "Limitaciones y próximos pasos")
bullets(s, [
    "Validación externa (ADNI), aumentar N",
    "Recalibración periódica si ECE > 0.2",
    "Explorar 2.5D/3D + explicabilidad (SHAP/Grad-CAM)",
])
notes(s, "Cerrar con plan realista de mejora y adopción clínica responsable.")

# ======= Slide 9 — Cierre =======
s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s)
title(s, "Gracias", "Demo: Streamlit / FastAPI  •  Contacto: equipo@cognitiva.ai")
notes(s, "Preguntas. Añadir QR a demo si está desplegada.")

out = "CognitivaAI_Deck_Draft.pptx"
prs.save(out)
print(f"✅ Listo: {out}")
