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
OUT_SUM = HERE / 'simulation_summary_5000.txt'

N = 5000

# Cargar diccionario (solo para referencia opcional)
with open(DIC_PATH, 'r', encoding='utf-8') as f:
    dic = json.load(f)

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
horas_trabajo_semana = clamp(horas_trabajo_semana, 0, 80)

# Horas de estudio correlates with rendimiento (desired corr ~0.6)
# We'll generate study_hours with moderate variance
study_base = np.random.normal(loc=12, scale=6, size=N)
study_base = clamp(study_base, 0, 80)

# Sleep hours
horas_sueno = np.random.normal(loc=7.0, scale=1.2, size=N)
horas_sueno = clamp(horas_sueno, 3.0, 10.0)

# Stress (0-100) influenced by career, work hours, and low sleep
estres = 30 + (horas_trabajo_semana * 0.9) - (horas_sueno * 4) + career_stress_shift + np.random.normal(0,10,size=N)
estres = clamp(estres, 0, 100)

# Apoyo social (0-100) correlated with tipo_fin and padres conditions
apoyo_social = 60 + np.random.normal(0,15,size=N) - (tipo_fin=='Pago_directo')*5
apoyo_social = clamp(apoyo_social, 0, 100)

# Autoeficacia (0-100) will be computed below after we calculate procrastination

# Procrastination index 0-100: inversely related to study hours and autoeficacia
indice_procrastinacion = clamp(60 - (study_base*1.8) + np.random.normal(0,12,size=N), 0, 100)

# Now compute autoeficacia properly using indice_procrastinacion
autoeficacia = clamp(40 + (apoyo_social*0.15) + (study_base*0.9) - (indice_procrastinacion*0.2) + np.random.normal(0,6,size=N), 0, 100)

# Horas deporte, cultura, extracurriculares
horas_deporte = clamp(np.random.poisson(1.2, size=N), 0, 30)
horas_cultura = clamp(np.random.poisson(0.8, size=N), 0, 30)
horas_extracurriculares = horas_deporte + horas_cultura + np.random.poisson(0.5, size=N)

# Entregas puntuales % correlated with procrastination and estudio
entregas_tarea_puntual = clamp(80 - (indice_procrastinacion*0.4) + (study_base*0.8) + np.random.normal(0,8,size=N), 0, 100)

# Horas pantalla no academica negatively impacts estudio modestly
horas_pantalla_no_academica = clamp(np.random.normal(3.0,1.8,size=N) + (indice_procrastinacion*0.02), 0, 12)

# Alimentacion index 0-100
alimentacion = clamp(50 + np.random.normal(0,12,size=N) + (ingresos_num/1000)*10, 0, 100)

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
pps = clamp(np.round(pps,2), 1.0, 20.0)

# pps_anterior slightly correlated with current
pps_anterior = clamp(np.round(pps + np.random.normal(0,1.6,size=N) - np.random.normal(0,0.6,size=N),2), 1.0, 20.0)

# rendimiento index composed from pps, entregas, asistencia (we don't have asistencia perc in this sim; approximate)
asistencia_total_pct = clamp(70 + np.random.normal(0,12,size=N) + (study_base*0.6) - (horas_trabajo_semana*0.3), 0, 100)
rendimiento = clamp( (pps/20)*100*0.5 + entregas_tarea_puntual*0.25 + asistencia_total_pct*0.25 + (autoeficacia*0.1), 0, 100)

# indice_financiero composite from ingresos, deuda, puntualidad
indice_financiero = clamp( (ingresos_num/ingresos_num.max())*100*0.5 + (1-has_debt)*100*0.3 + (puntualidad_pago)*100*0.2 + np.random.normal(0,6,size=N), 0, 100)

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
        pps[out_idx] = clamp(pps[out_idx] + np.random.choice([-6,6,8,10], size=out_idx.shape[0]), 1, 20)
    elif var == 'pps_anterior':
        pps_anterior[out_idx] = clamp(pps_anterior[out_idx] + np.random.choice([-6,6,8,10], size=out_idx.shape[0]), 1, 20)
    elif var == 'study_base':
        study_base[out_idx] = study_base[out_idx] + np.random.randint(15,50,size=out_idx.shape[0])
    elif var == 'horas_sueno':
        horas_sueno[out_idx] = clamp(horas_sueno[out_idx] + np.random.choice([-3,3,4], size=out_idx.shape[0]), 3, 12)
    elif var == 'estres':
        estres[out_idx] = clamp(estres[out_idx] + np.random.choice([20,30,-20], size=out_idx.shape[0]), 0, 100)
    elif var == 'entregas_tarea_puntual':
        entregas_tarea_puntual[out_idx] = clamp(entregas_tarea_puntual[out_idx] + np.random.choice([-40,40], size=out_idx.shape[0]), 0, 100)
    elif var == 'rendimiento':
        rendimiento[out_idx] = clamp(rendimiento[out_idx] + np.random.choice([-40,40], size=out_idx.shape[0]), 0, 100)
    elif var == 'indice_financiero':
        indice_financiero[out_idx] = clamp(indice_financiero[out_idx] + np.random.choice([-50,50], size=out_idx.shape[0]), 0, 100)
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

# Save CSV
OUT_DIR = HERE
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

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
