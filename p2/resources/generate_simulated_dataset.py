"""Generador reproducible de dataset simulado para Netica (Versión 11 - Lógica V9 Arreglada).
Salida:
 - p2/resources/simulated_students_500.csv (Archivo categórico principal)
 - p2/resources/simulation_summary_500.txt

Reglas implementadas (resumen):
 - N = 500 registros.
 - Generación numérica uniforme (para padres raíz).
 - Lógica de fórmulas (para hijos numéricos).
 - ¡CORREGIDO! Error 'KeyError' (V9/V10). Los nodos hijos (ej. asistencia_total_pct)
   ahora guardan sus puntajes en 'node_scores' para que los nodos nietos
   (ej. rendimiento) puedan usarlos.
 - Categorización Híbrida (Raíces Planas, Hijos Lógicos).
 - Eliminación de acentos/tildes para Netica.
 - Salida categórica con '_' (ej. 'Con_deuda').
"""

import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
import re
import traceback
import unicodedata # Para eliminar acentos

SEED = 42
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
N = 5000
DIC_PATH = HERE / 'diccionario' / 'diccionario_limpio.json'
OUT_CSV = HERE / f'simulated_students_{N}.csv'
OUT_NUMERIC = HERE / f'simulated_students_{N}_numeric.csv'
OUT_SUM = HERE / f'simulation_summary_{N}.txt'
OUT_CSV_CAT = HERE / f'simulated_students_{N}_categorical.csv'


# Cargar diccionario
with open(DIC_PATH, 'r', encoding='utf-8') as f:
    dic = json.load(f)

# --- Helper para limpiar texto para Netica ---
def normalize_text(s):
    """Elimina acentos, espacios y caracteres especiales."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s_norm = ''.join(c for c in unicodedata.normalize('NFD', s)
                     if unicodedata.category(c) != 'Mn')
    s_norm = s_norm.replace(' ', '_')
    s_norm = re.sub(r'[^A-Za-z0-9_]', '', s_norm)
    return s_norm

# --- Helpers de rangos ---
def parse_range_from_text(txt):
    if not isinstance(txt, str): return None
    txt = txt.strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", txt)
    if m: return float(m.group(1)), float(m.group(2))
    m = re.search(r">\s*(\d+(?:\.\d+)?)", txt)
    if m: return float(m.group(1)), None
    m = re.search(r"<\s*(\d+(?:\.\d+)?)", txt)
    if m: return None, float(m.group(1))
    m = re.search(r"^(\d+(?:\.\d+)?)$", txt)
    if m: v = float(m.group(1)); return v, v
    return None

def get_variable_range(var_name, dic_list):
    DEFAULT_CAPS = {'hours_week': 80, 'hours_day': 24, 'percentage': 100, 'pps': 20, 'alimentacion': 7, 'default': 100}
    entry = next((e for e in dic_list if e.get('Variable') == var_name), None)
    if not entry: return 0, DEFAULT_CAPS['default']
    como = entry.get('Como_se_mide', '')
    if 'Porcentaje' in como or '%' in como: return 0, DEFAULT_CAPS['percentage']
    if '1-20' in como: return 1, DEFAULT_CAPS['pps']
    if '1-7' in como: return 1, DEFAULT_CAPS['alimentacion']
    if 'Horas/día' in como: return 0, DEFAULT_CAPS['hours_day']
    if 'Horas/semana' in como: return 0, DEFAULT_CAPS['hours_week']
    cats = entry.get('Categorias') or []
    mins, maxs = [], []
    if isinstance(cats, list):
        for c in cats:
            rng = None
            if isinstance(c, dict) and 'rango' in c: rng = c['rango']
            elif isinstance(c, str): rng = c
            if rng:
                pr = parse_range_from_text(rng)
                if pr:
                    lo, hi = pr
                    if lo is not None: mins.append(lo)
                    if hi is not None: maxs.append(hi)
    lo, hi = (min(mins) if mins else None), (max(maxs) if maxs else None)
    var_lower = var_name.lower()
    if lo is None and hi is None:
        if 'pps' in var_lower: lo, hi = 1, DEFAULT_CAPS['pps']
        elif 'hora' in var_lower: lo, hi = 0, DEFAULT_CAPS['hours_week']
        elif 'indice' in var_lower: lo, hi = 0, DEFAULT_CAPS['percentage']
        elif 'alimentacion' in var_lower: lo, hi = 1, DEFAULT_CAPS['alimentacion']
        else: lo, hi = 0, DEFAULT_CAPS['default']
    if lo is None: lo = 0
    if hi is None:
        if 'pps' in var_lower: hi = DEFAULT_CAPS['pps']
        elif 'hora' in var_lower: hi = DEFAULT_CAPS['hours_week']
        elif 'sueno' in var_lower: hi = DEFAULT_CAPS['hours_day']
        elif 'alimentacion' in var_lower: hi = DEFAULT_CAPS['alimentacion']
        else: hi = DEFAULT_CAPS['default']
    return float(lo), float(hi)

ENFORCE_VARS = [e['Variable'] for e in dic]
VAR_RANGES = {v: get_variable_range(v, dic) for v in ENFORCE_VARS}
def clamp(arr, low, high):
    return np.minimum(np.maximum(arr, low), high)

# 1) Demográficos y estructura (Balanceados)
ids = np.arange(1, N+1)
sex = np.random.choice(['M','F','Other'], size=N, p=[0.48,0.49,0.03])
semestre = np.random.choice(np.arange(1,9), size=N, p=[0.18,0.18,0.16,0.14,0.12,0.10,0.07,0.05])
tipo_colegio = np.random.choice(['Publico','Concertado','Privado'], size=N, p=[0.55,0.25,0.20])
colegio_shift = np.where(tipo_colegio=='Privado', 1.2, np.where(tipo_colegio=='Concertado', 0.3, -0.6))
carreras = ['Ingenieria','Ciencias_Sociales','Salud','Artes','Negocios','Ciencias']
carrera = np.random.choice(carreras, size=N, p=[0.18,0.18,0.16,0.12,0.18,0.18])
career_stress_shift = np.isin(carrera, ['Ingenieria','Salud']).astype(float) * 10.0
tiene_hijos = np.random.choice([0,1], size=N, p=[0.93,0.07])
ingreso_cat = np.random.choice(['Muy_bajo','Bajo','Medio','Alto'], size=N, p=[0.25,0.35,0.30,0.10])
ingreso_median = np.array([300,600,1200,3000])
ingreso_map = dict(zip(['Muy_bajo','Bajo','Medio','Alto'], ingreso_median))
ingresos_num = np.array([ingreso_map[c] for c in ingreso_cat])
tf_entry = next(e for e in dic if e.get('Variable') == 'tipo_financiamiento')
tf_labels = [c if isinstance(c, str) else c.get('categoria') for c in tf_entry.get('Categorias', [])]
tipo_fin_labels_v1 = ['Beca_total','Beca_parcial','Credito','Financiamiento_familiar','Pago_directo']
tipo_fin = np.random.choice(tipo_fin_labels_v1, size=N, p=[0.08,0.12,0.20,0.30,0.30])
base_cost = ingresos_num * np.random.normal(0.12, 0.03, size=N)
costo_ratio = clamp(base_cost / ingresos_num, 0.05, 0.5)
costo_servicio_mensual = np.where(costo_ratio <= 0.1, 'Muy_bajo',
                           np.where(costo_ratio <= 0.2, 'Bajo',
                           np.where(costo_ratio <= 0.35, 'Medio','Alto')))

# 2) Generación Numérica (Raíces con Uniforme, Hijos con Lógica)
# (Esta es la lógica V1/V9, que SÍ funciona)

# --- Nodos Raíz (Uniforme) ---
ht_lo, ht_hi = VAR_RANGES.get('horas_trabajo_semana', (0,80))
horas_trabajo_semana = np.random.uniform(low=ht_lo, high=ht_hi, size=N)
sd_lo, sd_hi = VAR_RANGES.get('horas_de_estudio_semana', (0,80))
study_base = np.random.uniform(low=sd_lo, high=sd_hi, size=N)
hs_lo, hs_hi = VAR_RANGES.get('horas_sueno', (3.0,10.0))
horas_sueno = np.random.uniform(low=hs_lo, high=hs_hi, size=N)
as_lo, as_hi = VAR_RANGES.get('apoyo_social', (0,100))
apoyo_social = np.random.uniform(low=as_lo, high=as_hi, size=N)
apoyo_social = clamp(apoyo_social - (np.isin(tipo_fin, ['Pago_directo', 'Financiamiento_familiar']))*5, as_lo, as_hi)
hd_lo, hd_hi = VAR_RANGES.get('horas_deporte', (0,30))
horas_deporte = np.random.uniform(low=hd_lo, high=hd_hi, size=N)
hc_lo, hc_hi = VAR_RANGES.get('horas_cultura', (0,30))
horas_cultura = np.random.uniform(low=hc_lo, high=hc_hi, size=N)
hp_lo, hp_hi = VAR_RANGES.get('horas_pantalla_no_academica', (0,12))
horas_pantalla_no_academica = np.random.uniform(low=hp_lo, high=hp_hi, size=N)
al_lo, al_hi = VAR_RANGES.get('alimentacion', (0,100))
alimentacion = np.random.uniform(low=al_lo, high=al_hi, size=N)
atl_lo, atl_hi = VAR_RANGES.get('asistencia_laboratorio_pct', (0, 100))
asistencia_laboratorio_pct = np.random.uniform(atl_lo, atl_hi, size=N)
att_lo, att_hi = VAR_RANGES.get('asistencia_teorico_pct', (0, 100))
asistencia_teorico_pct = np.random.uniform(att_lo, att_hi, size=N)
phc_base = np.random.uniform(low=0, high=100, size=N)
phc_score = clamp(phc_base - (ingresos_num/1000)*15, 0, 100)

pp_entry = next(e for e in dic if e.get('Variable') == 'puntualidad_pago')
pp_cats = {c['categoria']: float(c['porcentaje'].replace('%',''))/100.0 for c in pp_entry['Categorias']}
puntualidad_label = np.random.choice(list(pp_cats.keys()), size=N, p=list(pp_cats.values()))
puntualidad_pago = (puntualidad_label == 'Puntual').astype(int)

da_entry = next(e for e in dic if e.get('Variable') == 'deuda_academica')
da_cats = {c['categoria']: float(c['porcentaje'].replace('%',''))/100.0 for c in da_entry['Categorias']}
deuda_label = np.random.choice(list(da_cats.keys()), size=N, p=list(da_cats.values()))
has_debt = (deuda_label == 'Con deuda').astype(int)

# --- Nodos Hijo (Lógica Numérica) ---
estres = 30 + (horas_trabajo_semana * 0.9) - (horas_sueno * 4) + career_stress_shift + np.random.uniform(-15, 15, size=N)
est_lo, est_hi = VAR_RANGES.get('estres', (0,100))
estres = clamp(estres, est_lo, est_hi)

ip_lo, ip_hi = VAR_RANGES.get('indice_procrastinacion', (0,100))
indice_procrastinacion = clamp(60 - (study_base*1.8) + np.random.uniform(-20, 20, size=N), ip_lo, ip_hi)

ae_lo, ae_hi = VAR_RANGES.get('autoeficacia_academica', (0,100))
autoeficacia = clamp(40 + (apoyo_social*0.15) + (study_base*0.9) - (indice_procrastinacion*0.2) + np.random.uniform(-10, 10, size=N), ae_lo, ae_hi)

hex_lo, hex_hi = VAR_RANGES.get('horas_extracurriculares', (0,80))
horas_extracurriculares = horas_deporte + horas_cultura + np.random.uniform(0, 5, size=N)
horas_extracurriculares = clamp(horas_extracurriculares, hex_lo, hex_hi)

et_lo, et_hi = VAR_RANGES.get('entregas_tarea_puntual', (0,100))
entregas_tarea_puntual = clamp(80 - (indice_procrastinacion*0.4) + (study_base*0.8) + np.random.uniform(-15, 15, size=N), et_lo, et_hi)

asistencia_total_pct = clamp( (asistencia_laboratorio_pct * 0.3) + (asistencia_teorico_pct * 0.7), 0, 100)

study_norm = (study_base - study_base.mean())/ (study_base.std()+1e-6)
sleep_norm = (horas_sueno - horas_sueno.mean())/(horas_sueno.std()+1e-6)
procr_norm = (indice_procrastinacion - indice_procrastinacion.mean())/(indice_procrastinacion.std()+1e-6)

# PPS y Rendimiento (Lógica V9)
# Esta lógica SÍ produce datos no planos, como en tu captura
pp_lo, pp_hi = VAR_RANGES.get('pps_actual_20', (1.0,20.0))
pps = 12 + (study_norm * 2.8) + (sleep_norm * 1.2) - (procr_norm * 2.0) + (colegio_shift * 0.6) + np.random.uniform(-3, 3, size=N)
pps = clamp(np.round(pps,2), pp_lo, pp_hi)

ppa_lo, ppa_hi = VAR_RANGES.get('pps_anterior_20', (1.0,20.0))
pps_anterior = clamp(np.round(pps + np.random.uniform(-2.5, 2.5, size=N),2), ppa_lo, ppa_hi)

rend_lo, rend_hi = VAR_RANGES.get('rendimiento', (0,100))
rendimiento = clamp( (pps/20)*100*0.5 + entregas_tarea_puntual*0.25 + asistencia_total_pct*0.25 + (autoeficacia*0.1), rend_lo, rend_hi)

ifn_lo, ifn_hi = VAR_RANGES.get('indice_financiero', (0,100))
indice_financiero = clamp( (ingresos_num/ingresos_num.max())*100*0.5 + (1-has_debt)*100*0.3 + (puntualidad_pago)*100*0.2 + np.random.uniform(-10, 10, size=N), ifn_lo, ifn_hi)

# Riesgo de Desercion y Bienestar (Lógica V9)
# (Estos no estaban en el V9, los añadimos)
base_prob = 0.5 - (rendimiento/200)
base_prob = clamp(base_prob, 0.02, 0.6)
mult = np.ones(N)
mult = mult * np.where(horas_trabajo_semana>20, 3.0, 1.0)
mult = mult * np.where(tipo_fin=='Beca_total', 0.5, 1.0)
mult = mult * np.where(estres>80, 2.5, 1.0)
# 'riesgo_de_desercion' será la categorización de 'prob_desercion'
prob_desercion = clamp(base_prob * mult, 0, 0.95)

# 'bienestar_estudiantil' (cálculo numérico)
habitos_num = (alimentacion/100.0) + (1-(horas_pantalla_no_academica/hp_hi)) + (horas_sueno/hs_hi)
bienestar_estudiantil = clamp( (habitos_num*0.3) + (horas_extracurriculares/hex_hi)*0.2 + (1-(estres/100.0))*0.3 + (apoyo_social/100.0)*0.2, 0, 1)


# Build Numeric DataFrame
df = pd.DataFrame({
    'id': ids,
    'sex': sex, 'semestre': semestre, 'carrera': carrera, 'tipo_colegio': tipo_colegio,
    'pps_actual_20': np.round(pps,2),
    'pps_anterior_20': np.round(pps_anterior,2),
    'horas_de_estudio_semana': np.round(study_base,2),
    'horas_trabajo_semana': np.round(horas_trabajo_semana,2),
    'horas_deporte': np.round(horas_deporte,2),
    'horas_cultura': np.round(horas_cultura,2),
    'horas_extracurriculares': np.round(horas_extracurriculares,2),
    'horas_sueno': np.round(horas_sueno,2),
    'estres': np.round(estres,2),
    'apoyo_social': np.round(apoyo_social,2),
    'autoeficacia_academica': np.round(autoeficacia,2),
    'entregas_tarea_puntual': np.round(entregas_tarea_puntual,2),
    'ingresos_mensual_cat': ingreso_cat,
    'tipo_financiamiento': tipo_fin,
    'costo_servicio_mensual_cat': costo_servicio_mensual,
    'horas_pantalla_no_academica': np.round(horas_pantalla_no_academica,2),
    'deuda_academica_label': deuda_label,
    'puntualidad_pago_label': puntualidad_label,
    'indice_financiero': np.round(indice_financiero,2),
    'rendimiento': np.round(rendimiento,2),
    'asistencia_total_pct': np.round(asistencia_total_pct,2),
    'asistencia_teorico_pct': np.round(asistencia_teorico_pct,2),
    'asistencia_laboratorio_pct': np.round(asistencia_laboratorio_pct,2),
    'indice_procrastinacion': np.round(indice_procrastinacion,2),
    'alimentacion': np.round(alimentacion,2),
    'phc_score': np.round(phc_score,2),
    'tiene_hijos': tiene_hijos,
    'riesgo_de_desercion': prob_desercion * 100, # Convertir a 0-100
    'bienestar_estudiantil': bienestar_estudiantil * 100, # Convertir a 0-100
    'actividad_fisica': np.round(horas_deporte, 2)
})

assert not df.isnull().any().any()
OUT_DIR = HERE
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(OUT_NUMERIC, index=False)

# --- Create a fully categorical dataset (V11 - Lógica V9 Arreglada) ---

def get_allowed_labels(entry):
    cats = entry.get('Categorias') or []
    allowed = []
    for c in cats:
        label = None
        if isinstance(c, dict) and 'categoria' in c:
            label = str(c['categoria'])
        elif isinstance(c, str):
            label = c
        if label:
            allowed.append(normalize_text(label))
    return allowed

def categorize_series_using_qcut(series, entry):
    allowed_labels = get_allowed_labels(entry)
    if not allowed_labels:
        return series.astype(str).apply(normalize_text)
    num_categories = len(allowed_labels)
    if num_categories == 0:
        return series.astype(str).apply(normalize_text)
    try:
        cat_series = pd.qcut(series, q=num_categories, labels=allowed_labels, duplicates='drop')
        if cat_series.isnull().any():
            na_count = cat_series.isnull().sum()
            cat_series[cat_series.isnull()] = np.random.choice(allowed_labels, size=na_count)
        return cat_series.astype(str)
    except Exception as e:
        return pd.Series(np.random.choice(allowed_labels, size=series.size), index=series.index)

# --- Build categorical DataFrame ---
cat_df = pd.DataFrame()
processed_vars = set()
node_scores = {}

try:
    # --- BUCLE 1: Mapeos Especiales (No Numéricos) ---
    print("Procesando mapeos especiales...")
    special_mappings = {
        'deuda_academica': df['deuda_academica_label'].apply(normalize_text),
        'ingresos_mensual': df['ingresos_mensual_cat'].astype(str).str.replace('_',' ').str.title().apply(normalize_text),
        'tipo_financiamiento': df['tipo_financiamiento'].astype(str).str.replace('_',' ').str.title().apply(normalize_text),
        'costo_servicio_mensual': df['costo_servicio_mensual_cat'].astype(str).str.replace('_',' ').str.title().apply(normalize_text),
        'puntualidad_pago': df['puntualidad_pago_label'].apply(normalize_text),
        'sex': df['sex'].apply(normalize_text),
        'carrera': df['carrera'].apply(normalize_text),
        'tipo_colegio': df['tipo_colegio'].apply(normalize_text),
    }
    for var, series in special_mappings.items():
        entry = next((e for e in dic if e.get('Variable') == var), None)
        if entry:
            cat_df[var] = series
            processed_vars.add(var)
            labels = get_allowed_labels(entry)
            if labels:
                score_map = {label: i for i, label in enumerate(labels)}
                node_scores[var] = series.map(score_map).fillna(0)

    # --- BUCLE 2: Nodos Raíz Numéricos (Usan 'qcut') ---
    print("Procesando nodos raíz numéricos con 'qcut'...")

    # Esta es la lista V9, que coincide con tu data (raíces planas)
    root_nodes = [
        'pps_actual_20', 'pps_anterior_20',
        'asistencia_teorico_pct', 'asistencia_laboratorio_pct',
        'autoeficacia_academica', 'entregas_tarea_puntual',
        'horas_de_estudio_semana', 'indice_procrastinacion',
        'alimentacion', 'actividad_fisica',
        'horas_pantalla_no_academica', 'horas_sueno',
        'horas_deporte', 'horas_cultura',
        'horas_trabajo_semana', 'estres', 'apoyo_social',
        'indice_financiero'
    ]

    for var in root_nodes:
        if var in processed_vars or var not in df.columns:
            continue

        entry = next((e for e in dic if e.get('Variable') == var), None)
        if not entry:
            continue

        cat_series = categorize_series_using_qcut(df[var], entry)
        cat_df[var] = cat_series
        processed_vars.add(var)

        labels = get_allowed_labels(entry)
        score_map = {label: i for i, label in enumerate(labels)}
        node_scores[var] = cat_series.map(score_map).fillna(0)

    # --- BUCLE 3: Nodos Hijos (Lógica de Reglas) ---
    print("Procesando nodos hijos con lógica de reglas...")

    # 1. asistencia_total_pct
    var = 'asistencia_total_pct'
    if var not in processed_vars:
        entry = next(e for e in dic if e.get('Variable') == var)
        score = (node_scores['asistencia_teorico_pct'] + node_scores['asistencia_laboratorio_pct']) / 2
        cat_series = categorize_series_using_qcut(score, entry)
        cat_df[var] = cat_series
        labels = get_allowed_labels(entry)
        score_map = {label: i for i, label in enumerate(labels)}
        node_scores[var] = cat_series.map(score_map).fillna(0)
        processed_vars.add(var)

    # 2. horas_extracurriculares
    var = 'horas_extracurriculares'
    if var not in processed_vars:
        entry = next(e for e in dic if e.get('Variable') == var)
        score = node_scores['horas_deporte'] + node_scores['horas_cultura']
        cat_series = categorize_series_using_qcut(score, entry)
        cat_df[var] = cat_series
        labels = get_allowed_labels(entry)
        score_map = {label: i for i, label in enumerate(labels)}
        node_scores[var] = cat_series.map(score_map).fillna(0)
        processed_vars.add(var)

    # 3. habitos_vida (Lógica V7 Correcta)
    var = 'habitos_vida'
    if var not in processed_vars:
        entry = next(e for e in dic if e.get('Variable') == var)

        alim_map = {normalize_text('Inadecuada'): 0, normalize_text('Moderada'): 1, normalize_text('Adecuada'): 2}
        pant_map = {normalize_text('Alto'): 0, normalize_text('Moderado'): 1, normalize_text('Bajo'): 2}
        sueno_map = {normalize_text('Deficiente'): 0, normalize_text('Regular'): 1, normalize_text('Adecuado'): 2, normalize_text('Excesivo'): 1}

        alim_score = cat_df['alimentacion'].map(alim_map).fillna(1)
        pant_score = cat_df['horas_pantalla_no_academica'].map(pant_map).fillna(1)
        sueno_score = cat_df['horas_sueno'].map(sueno_map).fillna(1)

        score = alim_score + pant_score + sueno_score
        cat_series = categorize_series_using_qcut(score, entry)
        cat_df[var] = cat_series
        labels = get_allowed_labels(entry)
        score_map = {label: i for i, label in enumerate(labels)}
        node_scores[var] = cat_series.map(score_map).fillna(0)
        processed_vars.add(var)

    # 4. rendimiento
    var = 'rendimiento'
    if var not in processed_vars:
        entry = next(e for e in dic if e.get('Variable') == var)
        # ¡CORREGIDO! 'asistencia_total_pct' ahora existe en node_scores
        score = (
            node_scores['pps_actual_20'] * 0.3 +
            node_scores['asistencia_total_pct'] * 0.2 +
            node_scores['autoeficacia_academica'] * 0.2 +
            node_scores['entregas_tarea_puntual'] * 0.2 +
            node_scores['horas_de_estudio_semana'] * 0.1
        )
        cat_series = categorize_series_using_qcut(score, entry)
        cat_df[var] = cat_series
        labels = get_allowed_labels(entry)
        score_map = {label: i for i, label in enumerate(labels)}
        node_scores[var] = cat_series.map(score_map).fillna(0)
        processed_vars.add(var)

    # 5. riesgo_de_desercion
    var = 'riesgo_de_desercion'
    if var not in processed_vars:
        entry = next(e for e in dic if e.get('Variable') == var)

        ip_score = node_scores['indice_procrastinacion']
        if_score = node_scores['indice_financiero']
        ht_score = node_scores['horas_trabajo_semana']
        es_score = node_scores['estres']

        aa_labels = get_allowed_labels(next(e for e in dic if e.get('Variable') == 'autoeficacia_academica'))
        aa_score_inv = node_scores['autoeficacia_academica'].map({i: len(aa_labels)-1-i for i in range(len(aa_labels))}).fillna(0)

        as_labels = get_allowed_labels(next(e for e in dic if e.get('Variable') == 'apoyo_social'))
        as_score_inv = node_scores['apoyo_social'].map({i: len(as_labels)-1-i for i in range(len(as_labels))}).fillna(0)

        score = ip_score + if_score + ht_score + es_score + aa_score_inv + as_score_inv
        cat_series = categorize_series_using_qcut(score, entry)
        cat_df[var] = cat_series
        labels = get_allowed_labels(entry)
        score_map = {label: i for i, label in enumerate(labels)}
        node_scores[var] = cat_series.map(score_map).fillna(0)
        processed_vars.add(var)

    # 6. bienestar_estudiantil
    var = 'bienestar_estudiantil'
    if var not in processed_vars:
        entry = next(e for e in dic if e.get('Variable') == var)

        hv_score = node_scores['habitos_vida']
        he_score = node_scores['horas_extracurriculares']
        as_score = node_scores['apoyo_social']

        es_labels = get_allowed_labels(next(e for e in dic if e.get('Variable') == 'estres'))
        es_score_inv = node_scores['estres'].map({i: len(es_labels)-1-i for i in range(len(es_labels))}).fillna(0)

        score = hv_score + he_score + es_score_inv + as_score
        cat_series = categorize_series_using_qcut(score, entry)
        cat_df[var] = cat_series
        labels = get_allowed_labels(entry)
        score_map = {label: i for i, label in enumerate(labels)}
        node_scores[var] = cat_series.map(score_map).fillna(0)
        processed_vars.add(var)


    # --- Finalización y Limpieza ---
    print("Finalizando y guardando...")
    cat_df.insert(0,'id', df['id'])
    dict_vars = [entry.get('Variable') for entry in dic if entry.get('Variable')]
    ordered_cols = ['id'] + [v for v in dict_vars if v != 'id']
    for c in cat_df.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    cat_df = cat_df.reindex(columns=[c for c in ordered_cols if c in cat_df.columns])

    for var in dict_vars:
        if var not in cat_df.columns and var != 'id':
            entry = next((e for e in dic if e.get('Variable') == var), None)
            if not entry: continue
            allowed = get_allowed_labels(entry)
            if allowed:
                cat_df[var] = pd.Series(np.random.choice(allowed, size=N), index=df.index)
            else:
                cat_df[var] = 'N_A'

    for col in cat_df.columns:
        if col == 'id': continue
        cat_df[col] = cat_df[col].astype(str).fillna('Unknown').apply(normalize_text)

    print(f'Saving categorical CSV to: {OUT_CSV_CAT}')
    cat_df.to_csv(OUT_CSV_CAT, index=False)
    print(f'Also saving categorical CSV as primary output: {OUT_CSV}')
    cat_df.to_csv(OUT_CSV, index=False)
    print('Categorical CSVs (compatibles con Netica) guardados.')
except Exception as e:
    err_path = OUT_DIR / 'categorical_error.txt'
    with open(err_path, 'w', encoding='utf-8') as ef:
        ef.write('Error building categorical dataset:\n')
        ef.write(traceback.format_exc())
    print(f'Error occurred while building categorical dataset. See {err_path}')
    raise e

# Create summary EDA
corr_cols = ['pps_actual_20','horas_de_estudio_semana','horas_trabajo_semana','estres','rendimiento', 'riesgo_de_desercion', 'bienestar_estudiantil']
correlations = df[corr_cols].corr()
with open(OUT_SUM, 'w', encoding='utf-8') as f:
    f.write('Correlaciones (Pearson) entre columnas clave (DATOS NUMÉRICOS INTERNOS):\n\n')
    f.write(str(correlations))
    f.write('\n\n')

print('Simulación completada:')
print(f' CSV Categórico (para Netica): {OUT_CSV}')
print(f' CSV Numérico (referencia): {OUT_NUMERIC}')
print(f' Summary: {OUT_SUM}')
