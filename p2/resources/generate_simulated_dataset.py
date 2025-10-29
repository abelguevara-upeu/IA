"""Generador reproducible de dataset simulado realista para estudiantes.
Salida:
 - p2/resources/simulated_students_5000.csv
 - p2/resources/simulation_summary_5000.txt

Reglas implementadas (resumen):
 - 5.000 registros
 - Relaciones y problemáticas solicitadas (trabajo, becas, sueño, estrés, semestres, tipo colegio)
 - 5% outliers en variables numéricas
 - Datos limpios y listos para entrenamiento (no NaNs)
"""

import json
import numpy as np
import pandas as pd
import os
from pathlib import Path

SEED = 42
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
DIC_PATH = HERE / 'diccionario' / 'diccionario_limpio.json'
OUT_CSV = HERE / 'simulated_students_5000.csv'
OUT_NUMERIC = HERE / 'simulated_students_5000_numeric.csv'
OUT_SUM = HERE / 'simulation_summary_5000.txt'
OUT_CSV_CAT = HERE / 'simulated_students_5000_categorical.csv'

N = 5000

# Cargar diccionario (solo para referencia opcional)
with open(DIC_PATH, 'r', encoding='utf-8') as f:
    dic = json.load(f)

# --- Helpers to extract numeric ranges from the immutable dictionary ---
import re
import traceback

def parse_range_from_text(txt):
    """Parse a range string like '0-7', '>18', '<2' or return numeric tokens found."""
    if not isinstance(txt, str):
        return None
    txt = txt.strip()
    # direct a-b
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", txt)
    if m:
        return float(m.group(1)), float(m.group(2))
    # >x
    m = re.search(r">\s*(\d+(?:\.\d+)?)", txt)
    if m:
        lo = float(m.group(1))
        return lo, None
    # <x
    m = re.search(r"<\s*(\d+(?:\.\d+)?)", txt)
    if m:
        hi = float(m.group(1))
        return None, hi
    # single number
    m = re.search(r"^(\d+(?:\.\d+)?)$", txt)
    if m:
        v = float(m.group(1))
        return v, v
    return None

def get_variable_range(var_name, dic_list):
    """Return a (min,max,default_cap) tuple for a variable based on the dictionary.
    If the dictionary provides open-ended ranges (None), we apply sensible caps depending on the measurement.
    """
    # sensible defaults per measurement type
    DEFAULT_CAPS = {
        'hours_week': 80,
        'hours_day': 24,
        'percentage': 100,
        'pps': 20,
        'alimentacion': 7,
        'default': 100
    }

    entry = next((e for e in dic_list if e.get('Variable') == var_name), None)
    if not entry:
        return 0, DEFAULT_CAPS['default']

    # If Como_se_mide mentions percentage or scale or specific bounds, prefer that
    como = entry.get('Como_se_mide', '')
    if 'Porcentaje' in como or '%' in como:
        return 0, DEFAULT_CAPS['percentage']
    if '1-20' in como or '1-20' in entry.get('Descripcion',''):
        return 1, DEFAULT_CAPS['pps']
    if '1-7' in como or '1-7' in entry.get('Descripcion',''):
        return 1, DEFAULT_CAPS['alimentacion']
    if 'Horas/día' in como or 'Horas/dia' in como:
        return 0, DEFAULT_CAPS['hours_day']
    if 'Horas/semana' in como:
        return 0, DEFAULT_CAPS['hours_week']

    cats = entry.get('Categorias') or []
    mins = []
    maxs = []
    if isinstance(cats, list):
        for c in cats:
            # each category could be a dict with 'rango' or a simple string
            rng = None
            if isinstance(c, dict) and 'rango' in c:
                rng = c['rango']
            elif isinstance(c, str):
                rng = c
            if rng:
                pr = parse_range_from_text(rng)
                if pr:
                    lo, hi = pr
                    if lo is not None:
                        mins.append(lo)
                    if hi is not None:
                        maxs.append(hi)

    # derive min and max
    lo = min(mins) if mins else None
    hi = max(maxs) if maxs else None

    # fallback heuristics based on variable name
    var_lower = var_name.lower()
    if lo is None and hi is None:
        if 'pps' in var_lower:
            lo, hi = 1, DEFAULT_CAPS['pps']
        elif 'hora' in var_lower or 'horas' in var_lower:
            lo, hi = 0, DEFAULT_CAPS['hours_week']
        elif 'indice' in var_lower or 'estres' in var_lower or 'apoyo' in var_lower or 'rendimiento' in var_lower:
            lo, hi = 0, DEFAULT_CAPS['percentage']
        elif 'alimentacion' in var_lower:
            lo, hi = 1, DEFAULT_CAPS['alimentacion']
        else:
            lo, hi = 0, DEFAULT_CAPS['default']

    # If one side is open (None), plug a sensible cap
    if lo is None:
        if 'hora' in var_lower:
            lo = 0
        else:
            lo = 0
    if hi is None:
        # choose cap according to likely units
        if 'pps' in var_lower:
            hi = DEFAULT_CAPS['pps']
        elif 'hora' in var_lower:
            hi = DEFAULT_CAPS['hours_week']
        elif 'sueno' in var_lower or 'dia' in var_lower:
            hi = DEFAULT_CAPS['hours_day']
        elif 'alimentacion' in var_lower:
            hi = DEFAULT_CAPS['alimentacion']
        else:
            hi = DEFAULT_CAPS['default']

    return float(lo), float(hi)

# Precompute ranges for variables we will enforce
ENFORCE_VARS = ['pps_actual_20','pps_anterior_20','horas_de_estudio_semana','horas_sueno','estres',
                'entregas_tarea_puntual','indice_procrastinacion','alimentacion','horas_pantalla_no_academica',
                'apoyo_social','autoeficacia_academica','indice_financiero','horas_deporte','horas_cultura',
                'horas_extracurriculares','horas_trabajo_semana']

VAR_RANGES = {v: get_variable_range(v, dic) for v in ENFORCE_VARS}


# Helper truncation
def clamp(arr, low, high):
    return np.minimum(np.maximum(arr, low), high)

# 1) Demográficos y estructura
ids = np.arange(1, N+1)
sex = np.random.choice(['M','F','Other'], size=N, p=[0.48,0.49,0.03])
semestre = np.random.choice(np.arange(1,9), size=N, p=[0.18,0.18,0.16,0.14,0.12,0.10,0.07,0.05])

# Tipo de colegio (problemática 2: brecha por colegio)
tipo_colegio = np.random.choice(['Publico','Concertado','Privado'], size=N, p=[0.55,0.25,0.20])
# Map performance shift by colegio
colegio_shift = np.where(tipo_colegio=='Privado', 1.2, np.where(tipo_colegio=='Concertado', 0.3, -0.6))

# Carrera (to simulate stress differences)
carreras = ['Ingenieria','Ciencias_Sociales','Salud','Artes','Negocios','Ciencias']
carrera = np.random.choice(carreras, size=N, p=[0.18,0.18,0.16,0.12,0.18,0.18])
# High stress careers: Ingenieria, Salud
career_stress_shift = np.isin(carrera, ['Ingenieria','Salud']).astype(float) * 10.0

# Tiene hijos (low prevalence)
tiene_hijos = np.random.choice([0,1], size=N, p=[0.93,0.07])

# Financiero
# ingresos_mensual categories: map to numeric median for internal calculations (in arbitrary units)
ingreso_cat = np.random.choice(['Muy_bajo','Bajo','Medio','Alto'], size=N, p=[0.25,0.35,0.30,0.10])
ingreso_median = np.array([300,600,1200,3000])  # arbitrary local currency units
ingreso_map = dict(zip(['Muy_bajo','Bajo','Medio','Alto'], ingreso_median))
ingresos_num = np.array([ingreso_map[c] for c in ingreso_cat])

# tipo_financiamiento (affects desercion)
tipo_fin = np.random.choice(['Beca_total','Beca_parcial','Credito','Financiamiento_familiar','Pago_directo'], size=N, p=[0.08,0.12,0.20,0.30,0.30])

# costo_servicio_mensual as fraction of income with categories
# base cost proportional to income but with noise
base_cost = ingresos_num * np.random.normal(0.12, 0.03, size=N)
costo_ratio = clamp(base_cost / ingresos_num, 0.05, 0.5)
# map to categories
costo_servicio_mensual = np.where(costo_ratio <= 0.1, 'Muy_bajo',
                           np.where(costo_ratio <= 0.2, 'Bajo',
                           np.where(costo_ratio <= 0.35, 'Medio','Alto')))

# Horas de trabajo por semana
# most students low, some with high hours
horas_trabajo_semana = np.random.choice(np.arange(0,61), size=N, p=None)
# bias: many zeros, few high -> we'll construct distribution
p = np.random.rand(N)
horas_trabajo_semana = (np.where(p<0.5, np.random.poisson(4, N), np.random.poisson(12, N)))
# add tail
tail_idx = np.random.choice(N, size=int(0.07*N), replace=False)
horas_trabajo_semana[tail_idx] = horas_trabajo_semana[tail_idx] + np.random.randint(10,41,size=tail_idx.shape[0])
ht_lo, ht_hi = VAR_RANGES.get('horas_trabajo_semana', (0,80))
horas_trabajo_semana = clamp(horas_trabajo_semana, ht_lo, ht_hi)

# Horas de estudio correlates with rendimiento (desired corr ~0.6)
# We'll generate study_hours with moderate variance
study_base = np.random.normal(loc=12, scale=6, size=N)
sd_lo, sd_hi = VAR_RANGES.get('horas_de_estudio_semana', (0,80))
study_base = clamp(study_base, sd_lo, sd_hi)

# Sleep hours
horas_sueno = np.random.normal(loc=7.0, scale=1.2, size=N)
hs_lo, hs_hi = VAR_RANGES.get('horas_sueno', (3.0,10.0))
horas_sueno = clamp(horas_sueno, hs_lo, hs_hi)

# Stress (0-100) influenced by career, work hours, and low sleep
estres = 30 + (horas_trabajo_semana * 0.9) - (horas_sueno * 4) + career_stress_shift + np.random.normal(0,10,size=N)
est_lo, est_hi = VAR_RANGES.get('estres', (0,100))
estres = clamp(estres, est_lo, est_hi)

# Apoyo social (0-100) correlated with tipo_fin and padres conditions
apoyo_social = 60 + np.random.normal(0,15,size=N) - (tipo_fin=='Pago_directo')*5
as_lo, as_hi = VAR_RANGES.get('apoyo_social', (0,100))
apoyo_social = clamp(apoyo_social, as_lo, as_hi)

# Autoeficacia (0-100) will be computed below after we calculate procrastination

# Procrastination index 0-100: inversely related to study hours and autoeficacia
ip_lo, ip_hi = VAR_RANGES.get('indice_procrastinacion', (0,100))
indice_procrastinacion = clamp(60 - (study_base*1.8) + np.random.normal(0,12,size=N), ip_lo, ip_hi)

# Now compute autoeficacia properly using indice_procrastinacion
ae_lo, ae_hi = VAR_RANGES.get('autoeficacia_academica', (0,100))
autoeficacia = clamp(40 + (apoyo_social*0.15) + (study_base*0.9) - (indice_procrastinacion*0.2) + np.random.normal(0,6,size=N), ae_lo, ae_hi)

# Horas deporte, cultura, extracurriculares
hd_lo, hd_hi = VAR_RANGES.get('horas_deporte', (0,30))
hc_lo, hc_hi = VAR_RANGES.get('horas_cultura', (0,30))
hex_lo, hex_hi = VAR_RANGES.get('horas_extracurriculares', (0,80))
horas_deporte = clamp(np.random.poisson(1.2, size=N), hd_lo, hd_hi)
horas_cultura = clamp(np.random.poisson(0.8, size=N), hc_lo, hc_hi)
horas_extracurriculares = horas_deporte + horas_cultura + np.random.poisson(0.5, size=N)
horas_extracurriculares = clamp(horas_extracurriculares, hex_lo, hex_hi)

# Entregas puntuales % correlated with procrastination and estudio
et_lo, et_hi = VAR_RANGES.get('entregas_tarea_puntual', (0,100))
entregas_tarea_puntual = clamp(80 - (indice_procrastinacion*0.4) + (study_base*0.8) + np.random.normal(0,8,size=N), et_lo, et_hi)

# Horas pantalla no academica negatively impacts estudio modestly
hp_lo, hp_hi = VAR_RANGES.get('horas_pantalla_no_academica', (0,12))
horas_pantalla_no_academica = clamp(np.random.normal(3.0,1.8,size=N) + (indice_procrastinacion*0.02), hp_lo, hp_hi)

# Alimentacion index 0-100
al_lo, al_hi = VAR_RANGES.get('alimentacion', (0,100))
alimentacion = clamp(50 + np.random.normal(0,12,size=N) + (ingresos_num/1000)*10, al_lo, al_hi)

# Calidad de vivienda / condiciones padres (PBC/PHC)
phc_score = clamp(50 + np.random.normal(0,20,size=N) - (ingresos_num/1000)*15, 0, 100)

# Deuda academica binary and monto
has_debt = np.random.binomial(1, p=np.where(ingreso_cat=='Muy_bajo',0.6, np.where(ingreso_cat=='Bajo',0.45, np.where(ingreso_cat=='Medio',0.25, 0.08))))
# debt amount in months of fee (approx)
debt_months = has_debt * (np.random.choice([1,2,3,6,12], size=N, p=[0.5,0.2,0.15,0.1,0.05]))

# Puntualidad pago as 0/1 based on has_debt and tipo_fin
puntualidad_pago = np.where(has_debt==1, np.random.choice([0,1], size=N, p=[0.6,0.4]), np.random.choice([0,1], size=N, p=[0.12,0.88]))
# map to categories
puntualidad_label = np.where(puntualidad_pago==1,'Puntual','Impuntual')

# Build pps_actual_20 influenced by study hours, sleep, procrastination, colegio_shift, and randomness
# Normalized factors
study_norm = (study_base - study_base.mean())/ (study_base.std()+1e-6)
sleep_norm = (horas_sueno - horas_sueno.mean())/(horas_sueno.std()+1e-6)
procr_norm = (indice_procrastinacion - indice_procrastinacion.mean())/(indice_procrastinacion.std()+1e-6)

pps = 12 + (study_norm * 2.8) + (sleep_norm * 1.2) - (procr_norm * 2.0) + (colegio_shift * 0.6) + np.random.normal(0,1.8,size=N)
pp_lo, pp_hi = VAR_RANGES.get('pps_actual_20', (1.0,20.0))
pps = clamp(np.round(pps,2), pp_lo, pp_hi)

# pps_anterior slightly correlated with current
ppa_lo, ppa_hi = VAR_RANGES.get('pps_anterior_20', (1.0,20.0))
pps_anterior = clamp(np.round(pps + np.random.normal(0,1.6,size=N) - np.random.normal(0,0.6,size=N),2), ppa_lo, ppa_hi)

# rendimiento index composed from pps, entregas, asistencia (we don't have asistencia perc in this sim; approximate)
asistencia_total_pct = clamp(70 + np.random.normal(0,12,size=N) + (study_base*0.6) - (horas_trabajo_semana*0.3), 0, 100)
rend_lo, rend_hi = VAR_RANGES.get('rendimiento', (0,100))
rendimiento = clamp( (pps/20)*100*0.5 + entregas_tarea_puntual*0.25 + asistencia_total_pct*0.25 + (autoeficacia*0.1), rend_lo, rend_hi)

# indice_financiero composite from ingresos, deuda, puntualidad
ifn_lo, ifn_hi = VAR_RANGES.get('indice_financiero', (0,100))
indice_financiero = clamp( (ingresos_num/ingresos_num.max())*100*0.5 + (1-has_debt)*100*0.3 + (puntualidad_pago)*100*0.2 + np.random.normal(0,6,size=N), ifn_lo, ifn_hi)

# Desercion probability base influenced by rendimiento, horas_trabajo, tipo_fin, estres, tiene hijos, semestre
# Base prob from low rendimiento
base_prob = 0.5 - (rendimiento/200)  # lower rendimento -> higher base prob
# adjust
base_prob = clamp(base_prob, 0.02, 0.6)

# multipliers
mult = np.ones(N)
mult = mult * np.where(horas_trabajo_semana>20, 3.0, 1.0)
mult = mult * np.where(tipo_fin=='Beca_total', 0.5, 1.0)
mult = mult * np.where(estres>80, 2.5, 1.0)
mult = mult * np.where(tiene_hijos==1, 1.5, 1.0)
# semestre effect
mult = mult * np.where(np.isin(semestre, [1,2,3,5,6]), 1.4, 1.0)

prob_desercion = clamp(base_prob * mult, 0, 0.95)
# sample desercion
desercion = np.random.binomial(1, prob_desercion)

# Apply 50% less prob for Beca_total already done via multiplier

# Add 5% outliers in numeric variables: pick indices and inject extremes
num_vars = ['pps','pps_anterior','study_base','horas_sueno','estres','entregas_tarea_puntual','rendimiento','indice_financiero','ingresos_num']
out_idx = np.random.choice(N, size=int(0.05*N), replace=False)
for var in num_vars:
    if var == 'pps':
        pps[out_idx] = clamp(pps[out_idx] + np.random.choice([-6,6,8,10], size=out_idx.shape[0]), pp_lo, pp_hi)
    elif var == 'pps_anterior':
        pps_anterior[out_idx] = clamp(pps_anterior[out_idx] + np.random.choice([-6,6,8,10], size=out_idx.shape[0]), ppa_lo, ppa_hi)
    elif var == 'study_base':
        study_base[out_idx] = study_base[out_idx] + np.random.randint(15,50,size=out_idx.shape[0])
        study_base = clamp(study_base, sd_lo, sd_hi)
    elif var == 'horas_sueno':
        horas_sueno[out_idx] = clamp(horas_sueno[out_idx] + np.random.choice([-3,3,4], size=out_idx.shape[0]), hs_lo, hs_hi)
    elif var == 'estres':
        estres[out_idx] = clamp(estres[out_idx] + np.random.choice([20,30,-20], size=out_idx.shape[0]), est_lo, est_hi)
    elif var == 'entregas_tarea_puntual':
        entregas_tarea_puntual[out_idx] = clamp(entregas_tarea_puntual[out_idx] + np.random.choice([-40,40], size=out_idx.shape[0]), et_lo, et_hi)
    elif var == 'rendimiento':
        rendimiento[out_idx] = clamp(rendimiento[out_idx] + np.random.choice([-40,40], size=out_idx.shape[0]), rend_lo, rend_hi)
    elif var == 'indice_financiero':
        indice_financiero[out_idx] = clamp(indice_financiero[out_idx] + np.random.choice([-50,50], size=out_idx.shape[0]), ifn_lo, ifn_hi)
    elif var == 'ingresos_num':
        ingresos_num[out_idx] = ingresos_num[out_idx] * np.random.choice([0.2,3,5], size=out_idx.shape[0])

# Normalize/round and make categorical labels consistent with dic
pps_actual_20 = np.round(pps,2)
pps_anterior_20 = np.round(pps_anterior,2)
horas_de_estudio_semana = np.round(study_base,2)
horas_sueno = np.round(horas_sueno,2)
estres = np.round(estres,2)
entregas_tarea_puntual = np.round(entregas_tarea_puntual,2)
rendimiento = np.round(rendimiento,2)
indice_financiero = np.round(indice_financiero,2)

# apoyo_social, autoeficacia already numeric
apoyo_social = np.round(apoyo_social,2)
autoeficacia = np.round(autoeficacia,2)

# Build DataFrame
df = pd.DataFrame({
    'id': ids,
    'sex': sex,
    'semestre': semestre,
    'carrera': carrera,
    'tipo_colegio': tipo_colegio,
    'pps_actual_20': pps_actual_20,
    'pps_anterior_20': pps_anterior_20,
    'horas_de_estudio_semana': horas_de_estudio_semana,
    'horas_trabajo_semana': horas_trabajo_semana,
    'horas_deporte': horas_deporte,
    'horas_cultura': horas_cultura,
    'horas_extracurriculares': horas_extracurriculares,
    'horas_sueno': horas_sueno,
    'estres': estres,
    'apoyo_social': apoyo_social,
    'autoeficacia_academica': autoeficacia,
    'entregas_tarea_puntual': entregas_tarea_puntual,
    'ingresos_mensual_cat': ingreso_cat,
    'ingresos_mensual': ingresos_num,
    'tipo_financiamiento': tipo_fin,
    'costo_servicio_mensual_cat': costo_servicio_mensual,
    'horas_pantalla_no_academica': horas_pantalla_no_academica,
    'deuda_academica_flag': has_debt,
    'deuda_academica_months': debt_months,
    'puntualidad_pago': puntualidad_label,
    'indice_financiero': indice_financiero,
    'rendimiento': rendimiento,
    'asistencia_total_pct': np.round(asistencia_total_pct,2),
    'indice_procrastinacion': np.round(indice_procrastinacion,2),
    'alimentacion': np.round(alimentacion,2),
    'phc_score': np.round(phc_score,2),
    'tiene_hijos': tiene_hijos,
    'desercion': desercion
})

# Ensure no NaNs
assert not df.isnull().any().any()

# Save numeric CSV (kept for reference). The primary OUT_CSV will be the categorical dataset below.
OUT_DIR = HERE
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(OUT_NUMERIC, index=False)

# --- Create a fully categorical dataset according to the immutable dictionary ---
def parse_category_ranges(cat_list):
    """Return a list of (label, (lo,hi)) for categories that include 'rango'.
    If range is open-ended, hi or lo can be None.
    """
    out = []
    for c in cat_list:
        if isinstance(c, dict) and 'rango' in c:
            rng = c['rango']
            pr = parse_range_from_text(rng)
            label = c.get('categoria') or str(rng)
            if pr:
                out.append((label, pr))
        elif isinstance(c, str):
            # no explicit range, skip here
            out.append((c, None))
    return out

def categorize_series_using_entry(series, entry):
    cats = entry.get('Categorias') or []
    parsed = parse_category_ranges(cats)
    # build canonical allowed labels from entry
    allowed = []
    for c in cats:
        if isinstance(c, dict) and 'categoria' in c:
            allowed.append(str(c['categoria']).strip())
        elif isinstance(c, str):
            allowed.append(c.strip())

    def _closest_allowed(label):
        if not isinstance(label, str):
            return None
        a = label.strip().lower()
        for opt in allowed:
            if opt.strip().lower() == a:
                return opt
        # try partial match
        for opt in allowed:
            if opt.strip().lower() in a or a in opt.strip().lower():
                return opt
        # no match
        return None

    res = pd.Series(index=series.index, dtype=object)
    if any(p[1] is not None for p in parsed):
        # numeric ranges available
        for label, pr in parsed:
            if pr is None:
                continue
            lo, hi = pr
            if lo is None:
                mask = series <= hi
            elif hi is None:
                mask = series > lo
            else:
                mask = (series >= lo) & (series <= hi)
            res.loc[mask] = label
        # any remaining NA assign closest category by distance to centers
        na_idx = res[res.isnull()].index
        if len(na_idx)>0:
            centers = []
            labels = []
            for label, pr in parsed:
                if pr is None:
                    continue
                lo, hi = pr
                if lo is None:
                    center = hi - 1
                elif hi is None:
                    center = lo + 1
                else:
                    center = (lo+hi)/2.0
                centers.append(center)
                labels.append(label)
            vals = series.loc[na_idx].values
            centers = np.array(centers)
            for i,v in zip(na_idx, vals):
                idx = int(np.argmin(np.abs(centers - v)))
                res.at[i] = labels[idx]
        # ensure labels are canonical
        for lab in list(res.unique()):
            if lab is None:
                continue
            ca = _closest_allowed(str(lab))
            if ca is None:
                continue
            res.loc[res==lab] = ca
        return res.ffill()
    else:
        # categorical list only: try to map existing values (case-insensitive)
        allowed = [c if isinstance(c,str) else c.get('categoria') for c in cats]
        allowed = [a for a in allowed if a is not None]
        sstr = series.astype(str).str.strip()
        out = sstr.copy()
        for a in allowed:
            mask = sstr.str.lower() == str(a).lower()
            out.loc[mask] = a
        unm = out[~out.isin(allowed)]
        if len(allowed)>0 and len(unm)>0:
            # sample from allowed labels when value cannot be matched
            choices = np.random.choice(allowed, size=len(unm))
            out.loc[unm.index] = choices
        # finalize by mapping to canonical allowed labels
        out = out.map(lambda x: _closest_allowed(x) or x)
        return out

# Build categorical DataFrame with all variables from the immutable dictionary
cat_df = pd.DataFrame()
try:
    def get_allowed_labels(entry):
        cats = entry.get('Categorias') or []
        allowed = []
        for c in cats:
            if isinstance(c, dict) and 'categoria' in c:
                allowed.append(str(c['categoria']).strip())
            elif isinstance(c, str):
                allowed.append(c.strip())
        return allowed

    for entry in dic:
        var = entry.get('Variable')
        # handle some known derivations
        if var == 'asistencia_teorico_pct':
            vals = (df['asistencia_total_pct'] * np.random.uniform(0.45,0.6,size=N)).round(2)
            cat_df[var] = categorize_series_using_entry(vals, entry)
            continue
        if var == 'asistencia_laboratorio_pct':
            vals = (df['asistencia_total_pct'] - df['asistencia_total_pct']*np.random.uniform(0.1,0.35,size=N)).round(2)
            vals = clamp(vals,0,100)
            cat_df[var] = categorize_series_using_entry(vals, entry)
            continue
        if var == 'asistencia_total_pct':
            cat_df[var] = categorize_series_using_entry(df['asistencia_total_pct'], entry)
            continue
        if var == 'deuda_academica':
            s = df.get('deuda_academica_flag', df.get('deuda_academica', pd.Series(np.zeros(N,dtype=int))))
            mapped = s.astype(int)
            cat_df[var] = mapped.map({1: 'Con deuda', 0: 'Sin deuda'}).fillna('Sin deuda')
            continue
        if var == 'ingresos_mensual':
            s = df.get('ingresos_mensual_cat', df.get('ingreso_cat', pd.Series(['Medio']*N)))
            s2 = s.astype(str).str.replace('_',' ').str.replace('\\s+',' ',regex=True).str.strip()
            s2 = s2.replace({'Muy bajo':'Muy bajo','Muy_bajo':'Muy bajo'})
            cat_df[var] = s2.map(lambda x: x.title() if isinstance(x,str) else 'Medio')
            continue
        if var == 'tipo_financiamiento':
            s = df.get('tipo_financiamiento', pd.Series(['Financiamiento_familiar']*N))
            s2 = s.astype(str).str.replace('_',' ').str.title()
            cat_df[var] = s2
            continue
        if var == 'costo_servicio_mensual':
            s = df.get('costo_servicio_mensual_cat', df.get('costo_servicio_mensual', pd.Series(['Medio']*N)))
            s2 = s.astype(str).str.replace('_',' ').str.title()
            cat_df[var] = s2
            continue
        if var == 'puntualidad_pago':
            cat_df[var] = df['puntualidad_pago']
            continue
        if var == 'actividad_fisica':
            h = df.get('horas_deporte', pd.Series(np.zeros(N)))
            labels = pd.Series(index=h.index, dtype=object)
            labels.loc[h==0] = 'Nula'
            labels.loc[(h>=1)&(h<=2)] = 'Baja'
            labels.loc[(h>=3)&(h<=4)] = 'Frecuente'
            labels.loc[h>=5] = 'Constante'
            cat_df[var] = labels.fillna('Nula')
            continue
        if var == 'habitos_vida':
            a = df['alimentacion']
            p = df['horas_pantalla_no_academica']
            s = df['horas_sueno']
            score = ( (a/ a.max())*0.5 + (1 - (p / (p.max()+1)))*0.25 + (s/ (s.max()+1))*0.25 )
            # use dictionary allowed labels and split into 4 buckets if available
            allowed = get_allowed_labels(entry)
            if len(allowed) >= 4:
                labels = allowed[:4]
                bins = [-1, 0.25, 0.5, 0.75, 1.1]
                cat_df[var] = pd.cut(score, bins=bins, labels=labels).astype(object).fillna(labels[1])
            else:
                bins = [ -1, 0.33, 0.66, 1.1]
                labels = ['Deficiente','Regular','Bueno']
                cat_df[var] = pd.cut(score, bins=bins, labels=labels).astype(object).fillna('Regular')
            continue
        if var == 'horas_de_estudio_semana':
            cat_df[var] = categorize_series_using_entry(df['horas_de_estudio_semana'], entry)
            continue
        if var in df.columns:
            if np.issubdtype(df[var].dtype, np.number):
                cat_df[var] = categorize_series_using_entry(df[var], entry)
            else:
                # try to map string-like values to allowed categories
                # use categorize_series_using_entry which handles categorical lists
                cat_df[var] = categorize_series_using_entry(df[var], entry)
            continue
        # fallback: sample from allowed labels (try to respect 'porcentaje' if present)
        allowed = get_allowed_labels(entry)
        if len(allowed) == 0:
            # as a last resort, create a generic label
            sampled = ['N/A'] * N
        else:
            # look for porcentaje fields inside category dicts
            weights = None
            cats = entry.get('Categorias') or []
            pct_list = []
            for c in cats:
                if isinstance(c, dict) and 'porcentaje' in c:
                    v = c.get('porcentaje')
                    try:
                        pct_list.append(float(str(v).replace('%','').strip())/100.0)
                    except Exception:
                        pct_list.append(None)
                else:
                    pct_list.append(None)
            if any(x is not None for x in pct_list):
                # build weights aligned to allowed (fill missing with uniform)
                weights = []
                for i,a in enumerate(allowed):
                    w = pct_list[i] if i < len(pct_list) and pct_list[i] is not None else None
                    weights.append(w if w is not None else 1.0)
                w = np.array(weights, dtype=float)
                w = w / w.sum()
                sampled = list(np.random.choice(allowed, size=N, p=w))
            else:
                sampled = list(np.random.choice(allowed, size=N))
        cat_df[var] = pd.Series(sampled, index=df.index)

    for c in cat_df.columns:
        cat_df[c] = cat_df[c].fillna('Unknown')

    cat_df.insert(0,'id', df['id'])
    # Ensure all variables from the immutable dictionary are present in the categorical dataset
    dict_vars = [entry.get('Variable') for entry in dic if entry.get('Variable')]
    # Reorder columns: id first, then variables as in the dictionary (if present)
    ordered_cols = ['id'] + [v for v in dict_vars if v != 'id']
    # Add any missing columns that might have been generated to the end
    for c in cat_df.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    cat_df = cat_df.reindex(columns=[c for c in ordered_cols if c in cat_df.columns])

    # Coerce all columns (except id) to string/categorical values to guarantee no numeric columns remain
    for c in cat_df.columns:
        if c == 'id':
            continue
        # convert to string and strip
        cat_df[c] = cat_df[c].astype(str).str.strip()

    # Save categorical CSV to both the categorical-specific path and the primary OUT_CSV path
    print('Saving categorical CSV to:', OUT_CSV_CAT)
    cat_df.to_csv(OUT_CSV_CAT, index=False)
    print('Also saving categorical CSV as primary output:', OUT_CSV)
    cat_df.to_csv(OUT_CSV, index=False)
    print('Categorical CSVs saved.')
except Exception as e:
    err_path = OUT_DIR / 'categorical_error.txt'
    with open(err_path, 'w', encoding='utf-8') as ef:
        ef.write('Error building categorical dataset:\n')
        ef.write(traceback.format_exc())
    print('Error occurred while building categorical dataset. See', err_path)

# Create summary EDA
corr_cols = ['pps_actual_20','horas_de_estudio_semana','horas_trabajo_semana','estres','desercion']
correlations = df[corr_cols].corr()

# Desercion by conditions
rate_work20 = df.loc[df.horas_trabajo_semana>20, 'desercion'].mean()
rate_overall = df.desercion.mean()
rate_beca_total = df.loc[df.tipo_financiamiento=='Beca_total', 'desercion'].mean()
rate_sleep_bad = df.loc[df.horas_sueno<5, 'desercion'].mean()
rate_stress_high = df.loc[df.estres>80, 'desercion'].mean()

with open(OUT_SUM, 'w', encoding='utf-8') as f:
    f.write('Correlaciones (Pearson) entre columnas clave:\n\n')
    f.write(str(correlations))
    f.write('\n\n')
    f.write(f'Tasa de deserción general: {rate_overall:.3f}\n')
    f.write(f'Tasa de deserción (trabajo>20h): {rate_work20:.3f}\n')
    f.write(f'Tasa de deserción (beca_total): {rate_beca_total:.3f}\n')
    f.write(f'Tasa de deserción (sueño<5h): {rate_sleep_bad:.3f}\n')
    f.write(f'Tasa de deserción (estres>80): {rate_stress_high:.3f}\n')

print('Simulación completada:')
print(' CSV:', OUT_CSV)
print(' Summary:', OUT_SUM)
