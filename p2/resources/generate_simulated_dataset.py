"""
generate_simulated_dataset.py

Genera un dataset simulado de estudiantes (1,000 registros) para analizar
rendimiento académico, deserción y bienestar estudiantil.

Dependencias: pandas, numpy

Salida: /Users/abelguevarah/Desktop/IA/p2/resources/simulated_students_1000.csv

Uso: python generate_simulated_dataset.py

El script está comentado y es determinista usando una semilla.
"""
import os
import random
import json
import math
import numpy as np
import pandas as pd


def minmax(series):
    s = np.array(series, dtype=float)
    lo = s.min()
    hi = s.max()
    if hi == lo:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def generate(n=1000, seed=42, out_path=None, noise_level=0.05, outlier_frac=0.02, label_noise=0.02):
    """
    Genera el dataset.

    Parámetros adicionales para simular realismo sin introducir valores faltantes:
    - noise_level: fracción del desvío estándar añadida como ruido gaussiano a variables continuas.
    - outlier_frac: fracción de registros a convertir en outliers plausibles (valores extremos).
    - label_noise: fracción de etiquetas de deserción a volcar para simular medición/etiqueta imperfecta.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Cargar diccionario estructurado y validar relaciones padre-hijo
    dic_path = os.path.join(os.path.dirname(__file__), 'diccionario', 'diccionario_limpio.json')
    if os.path.exists(dic_path):
        with open(dic_path, 'r', encoding='utf-8') as f:
            dic = json.load(f)
    else:
        raise FileNotFoundError(f"Diccionario no encontrado en {dic_path}")

    dic_map = {v['Variable']: v for v in dic}

    # Validar que los padres referenciados existan (soporta múltiples padres en 'Padres')
    parent_names = set()
    for v in dic:
        for p in v.get('Padres', []):
            if p:
                parent_names.add(p)
    missing_parents = [p for p in sorted(parent_names) if p not in dic_map]
    if missing_parents:
        raise ValueError(f"Padres referenciados en el diccionario no existen: {missing_parents}")

    # Construir hijos por padre y verificar variables compuestas tienen hijos
    children_by_parent = {}
    for v in dic:
        for p in v.get('Padres', []):
            if p:
                children_by_parent.setdefault(p, []).append(v['Variable'])
    composed_vars = [v['Variable'] for v in dic if v.get('Tipo_variable') and str(v.get('Tipo_variable')).lower().startswith('comp')]
    missing_children = [cv for cv in composed_vars if cv not in children_by_parent or len(children_by_parent.get(cv, [])) == 0]
    if missing_children:
        raise ValueError(f"Variables compuestas sin hijos en el diccionario: {missing_children}")

    # Preparar categorías/probabilidades para tipo_financiamiento según diccionario
    tf_def = dic_map.get('tipo_financiamiento')
    if tf_def:
        raw_cats = tf_def.get('Categorias') or []
        # Si items son objetos con porcentaje, extraerlo
        financiamiento_cats = []
        financiamiento_probs = []
        for it in raw_cats:
            if isinstance(it, dict):
                financiamiento_cats.append(it.get('categoria'))
                pct = it.get('porcentaje')
                if pct:
                    try:
                        financiamiento_probs.append(float(str(pct).strip('%')) / 100.0)
                    except Exception:
                        financiamiento_probs.append(None)
                else:
                    financiamiento_probs.append(None)
            else:
                financiamiento_cats.append(it)
                financiamiento_probs.append(None)
        # Si no se especificaron probabilidades, asignar fallback razonable
        if all(p is None for p in financiamiento_probs):
            if set(['Padres', 'Mixto', 'Propio']).issubset(set(financiamiento_cats)):
                financiamiento_probs = [0.5 if c == 'Padres' else 0.3 if c == 'Mixto' else 0.2 if c == 'Propio' else 0.0 for c in financiamiento_cats]
                s = sum(financiamiento_probs)
                if s <= 0:
                    financiamiento_probs = [1.0 / len(financiamiento_cats)] * len(financiamiento_cats)
                else:
                    financiamiento_probs = [p / s for p in financiamiento_probs]
            else:
                financiamiento_probs = [1.0 / len(financiamiento_cats)] * len(financiamiento_cats) if financiamiento_cats else [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            # rellenar probabilidades faltantes y normalizar
            probs = [0.0 if p is None else p for p in financiamiento_probs]
            missing = sum(1 for p in financiamiento_probs if p is None)
            rem = max(0.0, 1.0 - sum(p for p in probs))
            if missing > 0:
                for i, p in enumerate(financiamiento_probs):
                    if p is None:
                        probs[i] = rem / missing
            # normalizar
            s = sum(probs)
            if s <= 0:
                probs = [1.0 / len(probs)] * len(probs)
            else:
                probs = [p / s for p in probs]
            financiamiento_probs = probs
    else:
        financiamiento_cats = ['Beca_total', 'Beca_parcial', 'Financiamiento_familiar', 'Pago_directo', 'Credito']
        financiamiento_probs = [0.08, 0.12, 0.45, 0.30, 0.05]

    # --- Parámetros razonables para simular poblaciones universitarias ---
    # Las categorías/probabilidades de tipo de financiamiento se definirán a partir
    # del diccionario estructurado cuando esté disponible más abajo.

    # Generación base
    tipo_fin = np.random.choice(financiamiento_cats, size=n, p=financiamiento_probs)
    edad = np.random.normal(loc=22, scale=3, size=n).astype(int)
    edad = np.clip(edad, 17, 60)

    # Hijos (evento poco frecuente, pero tiene impacto en deserción)
    tiene_hijos = np.random.binomial(1, 0.07, size=n)

    # Ingresos mensuales (unidad arbitraria, más bajo = menor capacidad)
    # Usamos una gamma para obtener distribuciones sesgadas a la derecha
    ingresos = np.round(np.random.gamma(shape=2.0, scale=300.0, size=n)).astype(int)

    # Horas de trabajo por semana: muchos no trabajan, algunos a tiempo parcial, pocos a tiempo completo
    trabaja_flag = np.random.binomial(1, 0.55, size=n)  # 55% realizan alguna actividad laboral
    horas_trabajo = np.where(trabaja_flag == 0, 0, np.random.poisson(lam=8, size=n))
    horas_trabajo = np.clip(horas_trabajo, 0, 60)

    # Horas de estudio por semana: correlacionadas negativamente con horas de trabajo
    base_estudio = np.random.normal(loc=12, scale=4, size=n)
    horas_de_estudio_semana = base_estudio - 0.25 * horas_trabajo + np.random.normal(0, 2, size=n)
    horas_de_estudio_semana = np.clip(horas_de_estudio_semana, 0, 80)

    # Horas de sueño por noche
    horas_sueno = np.random.normal(loc=7.0, scale=1.0, size=n) - 0.02 * horas_trabajo
    horas_sueno = np.clip(horas_sueno, 3.0, 10.0)

    # Actividades (deporte, cultura, extracurriculares) - pequeñas, sesgadas
    horas_deporte = np.random.choice([0, 1, 2, 3, 5], size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    horas_cultura = np.random.choice([0, 1, 2, 3, 5], size=n, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    horas_extracurriculares = np.round((horas_deporte + horas_cultura) * np.random.uniform(0.5, 1.2, size=n)).astype(int)

    # Procrastinación (0-100) y apoyo social (0-100)
    indice_procrastinacion = np.random.beta(a=2.0, b=3.0, size=n) * 100
    apoyo_social = np.clip(np.random.normal(loc=60, scale=15, size=n), 0, 100)

    # Deuda en meses (más probabilidad si ingresos bajos o sin financiamiento familiar)
    ingreso_norm = minmax(ingresos)
    deuda_meses = np.random.poisson(lam=np.clip(1.5 + (1 - ingreso_norm) * 2.5, 0.2, 6.0), size=n)

    # Puntualidad de pago: depende de ingresos y deuda
    prob_puntual = 0.9 - 0.6 * (deuda_meses > 0).astype(float) - 0.3 * (ingreso_norm < 0.3).astype(float)
    prob_puntual = np.clip(prob_puntual, 0.05, 0.98)
    puntualidad_pago = np.random.binomial(1, prob_puntual)

    # Indice financiero: compuesto (ingresos, deuda, puntualidad)
    ingresos_norm = minmax(ingresos)
    deuda_norm = minmax(deuda_meses)
    puntual_norm = puntualidad_pago  # ya binaria 0/1
    indice_financiero_score = 0.5 * ingresos_norm + (-0.3) * deuda_norm + 0.2 * puntual_norm
    # Normalizar a 0-100
    indice_financiero = np.clip(minmax(indice_financiero_score) * 100, 0, 100)

    # Estrés: función de horas de estudio (carga), horas de trabajo, poca cama, procrastinación, bajo apoyo
    estudio_norm = minmax(horas_de_estudio_semana)
    trabajo_norm = minmax(horas_trabajo)
    sueno_norm = minmax(horas_sueno)
    procrast_norm = minmax(indice_procrastinacion)

    estres_score = (
        0.35 * estudio_norm
        + 0.30 * trabajo_norm
        + 0.20 * procrast_norm
        - 0.40 * apoyo_social / 100.0
        - 0.25 * sueno_norm
    )
    # Añadir ruido y escalar
    estres = np.clip((minmax(estres_score) + np.random.normal(0, 0.05, size=n)) * 100, 0, 100)

    # Asistencia (teorica y laboratorio) correlacionadas con horas de estudio y trabajo (negatively)
    asistencia_teorico_pct = np.clip(80 + 20 * estudio_norm - 15 * trabajo_norm + np.random.normal(0, 5, size=n), 0, 100)
    asistencia_laboratorio_pct = np.clip(75 + 18 * estudio_norm - 10 * trabajo_norm + np.random.normal(0, 6, size=n), 0, 100)
    asistencia_total_pct = (0.6 * asistencia_teorico_pct + 0.4 * asistencia_laboratorio_pct)

    # Entregas puntuales correlacionan con procrastinación y apoyo social
    entregas_tarea_puntual = np.clip(100 - 40 * procrast_norm + 10 * (apoyo_social / 100.0) + np.random.normal(0, 8, size=n), 0, 100)

    # Autoeficacia relacionada con apoyo y procrastinación
    autoeficacia_academica = np.clip(40 + 30 * (apoyo_social / 100.0) - 20 * procrast_norm + np.random.normal(0, 7, size=n), 0, 100)

    # Calidad de sueño como categorica derivada
    horas_sueno_cat = []
    for hs in horas_sueno:
        if hs < 5:
            horas_sueno_cat.append('Deficiente')
        elif hs < 7:
            horas_sueno_cat.append('Regular')
        elif hs <= 8:
            horas_sueno_cat.append('Adecuado')
        else:
            horas_sueno_cat.append('Excesivo')

    # Promedio ponderado actual (pps_actual_20) entre 0 y 20
    # Base: 11-13, incrementado por horas de estudio y apoyo, reducido por trabajo y estrés
    pps_base = 11.5 + 5.5 * estudio_norm - 3.5 * trabajo_norm + 1.5 * (apoyo_social / 100.0) - 2.0 * (estres / 100.0)
    # Becas incrementan rendimiento ligeramente
    pps_bonus = np.where(tipo_fin == 'Beca_total', 1.2, np.where(tipo_fin == 'Beca_parcial', 0.6, 0.0))
    pps_actual_20 = np.clip(pps_base + pps_bonus + np.random.normal(0, 1.3, size=n), 0, 20)

    # Promedio anterior (algo correlacionado con actual pero con ruido)
    pps_anterior_20 = np.clip(pps_actual_20 + np.random.normal(0, 1.5, size=n), 0, 20)

    # Indices de comportamiento (alimentación, actividad física) - convertir a categorías simples
    alimentacion_score = np.clip(np.random.normal(60 - 10 * (1 - ingresos_norm), 12, size=n), 0, 100)
    actividad_fisica_days = np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])

    # Puntaje de 'PQS' o calidad de sueño (complementario a horas de sueño)
    pq_score = np.clip(100 - (50 * (np.array([{'Deficiente':2,'Regular':1,'Adecuado':0,'Excesivo':0}.get(x,0) for x in horas_sueno_cat])) ) + np.random.normal(0,10,size=n), 0,100)

    # Riesgo de deserción (probabilidad) -> mayor si pps bajo, trabaja mucho, tiene hijos, deuda, baja asistencia
    pps_norm = minmax(pps_actual_20)
    riesgo_score = (
        0.55 * (1 - pps_norm)
        + 0.25 * trabajo_norm
        + 0.12 * (debt_indicator := (deuda_meses > 0).astype(float))
        + 0.18 * tiene_hijos
        - 0.10 * (tipo_fin == 'Beca_total').astype(float)  # becados menos riesgo
        - 0.08 * (asistencia_total_pct / 100.0)
    )
    # Convertir a probabilidad entre 0 y 1 con logistic-like scaling
    riesgo_prob = 1 / (1 + np.exp(-((riesgo_score - np.mean(riesgo_score)) * 3)))
    desercion = np.random.binomial(1, riesgo_prob)

    # Mapear categorías resumen para variables que en el diccionario están categorizadas
    def categorize_hours_study(h):
        if h <= 5:
            return 'Insuficiente'
        elif h <= 10:
            return 'Limitado'
        elif h <= 18:
            return 'Adecuado'
        else:
            return 'Intensivo'

    horas_estudio_cat = [categorize_hours_study(h) for h in horas_de_estudio_semana]

    # Construir DataFrame
    df = pd.DataFrame({
        'edad': edad,
        'tiene_hijos': tiene_hijos,
        'tipo_financiamiento': tipo_fin,
        'ingresos_mensual': ingresos,
        'deuda_meses': deuda_meses,
        'puntualidad_pago': puntualidad_pago,
        'indice_financiero': np.round(indice_financiero, 1),
        'horas_trabajo_semana': horas_trabajo,
        'horas_de_estudio_semana': np.round(horas_de_estudio_semana, 1),
        'horas_sueno': np.round(horas_sueno, 2),
        'horas_deporte': horas_deporte,
        'horas_cultura': horas_cultura,
        'horas_extracurriculares': horas_extracurriculares,
        'actividad_fisica': actividad_fisica_days,
        'alimentacion_score': np.round(alimentacion_score, 1),
        'indice_procrastinacion': np.round(indice_procrastinacion, 1),
        'apoyo_social': np.round(apoyo_social, 1),
        'estres': np.round(estres, 1),
        'asistencia_teorico_pct': np.round(asistencia_teorico_pct, 1),
        'asistencia_laboratorio_pct': np.round(asistencia_laboratorio_pct, 1),
        'asistencia_total_pct': np.round(asistencia_total_pct, 1),
        'entregas_tarea_puntual': np.round(entregas_tarea_puntual, 1),
        'autoeficacia_academica': np.round(autoeficacia_academica, 1),
        'pps_actual_20': np.round(pps_actual_20, 2),
        'pps_anterior_20': np.round(pps_anterior_20, 2),
        'horas_sueno_cat': horas_sueno_cat,
        'pps_categoria_estudio': horas_estudio_cat,
        'desercion': desercion,
        'debt_indicator': debt_indicator.astype(int),
    })

    # ---------------------- Introducir ruido controlado ----------------------
    # Añadir ruido gaussiano relativo al std de cada columna numérica pero mantener tipos coherentes
    cont_cols = [
        'ingresos_mensual', 'deuda_meses', 'indice_financiero', 'horas_trabajo_semana',
        'horas_de_estudio_semana', 'horas_sueno', 'horas_deporte', 'horas_cultura',
        'horas_extracurriculares', 'actividad_fisica', 'alimentacion_score',
        'indice_procrastinacion', 'apoyo_social', 'estres', 'asistencia_teorico_pct',
        'asistencia_laboratorio_pct', 'asistencia_total_pct', 'entregas_tarea_puntual',
        'autoeficacia_academica', 'pps_actual_20', 'pps_anterior_20', 'costo_servicio_mensual'
    ]

    # Asegurar columna costo_servicio_mensual existe
    if 'costo_servicio_mensual' not in df.columns:
        df['costo_servicio_mensual'] = np.round(np.clip(df['ingresos_mensual'] * np.random.uniform(0.05, 0.3, size=n), 10, None), 0)

    for col in cont_cols:
        if col in df.columns:
            std = max(df[col].std(), 1e-6)
            noise = np.random.normal(0, noise_level * std, size=n)
            # aplicar y recortar para mantener rangos plausibles
            df[col] = df[col] + noise
            if df[col].dtype.kind in 'fi':
                # recortar negatividades donde no aplican
                if col in ['ingresos_mensual', 'deuda_meses', 'horas_trabajo_semana', 'horas_deporte', 'actividad_fisica', 'costo_servicio_mensual']:
                    df[col] = np.clip(df[col], 0, None)

    # Reducir decimales innecesarios: aplicar reglas por tipo (evita muchos decimales poco creíbles)
    round_map = {
        'ingresos_mensual': 0,
        'deuda_meses': 0,
        'indice_financiero': 1,
        'horas_trabajo_semana': 0,
        'horas_de_estudio_semana': 1,
        'horas_sueno': 2,
        'horas_deporte': 0,
        'horas_cultura': 0,
        'horas_extracurriculares': 0,
        'actividad_fisica': 0,
        'alimentacion_score': 1,
        'indice_procrastinacion': 1,
        'apoyo_social': 1,
        'estres': 1,
        'asistencia_teorico_pct': 1,
        'asistencia_laboratorio_pct': 1,
        'asistencia_total_pct': 1,
        'entregas_tarea_puntual': 1,
        'autoeficacia_academica': 1,
        'pps_actual_20': 2,
        'pps_anterior_20': 2,
        'costo_servicio_mensual': 0
    }
    for col, digs in round_map.items():
        if col in df.columns:
            if digs == 0:
                # convertir a entero donde corresponde
                df[col] = np.round(df[col]).astype(int)
            else:
                df[col] = df[col].round(digs)

    # ---------------------- Introducir outliers plausibles ----------------------
    n_out = int(max(1, round(outlier_frac * n)))
    if n_out > 0:
        out_idx = np.random.choice(df.index, size=n_out, replace=False)
        # seleccionar algunas columnas para acentuar en outliers
        out_cols = ['ingresos_mensual', 'deuda_meses', 'horas_trabajo_semana', 'pps_actual_20']
        for idx in out_idx:
            for col in out_cols:
                if col in df.columns:
                    factor = np.random.choice([1.8, 2.5, 3.5])
                    # Aumentar o disminuir drásticamente según columna
                    if col in ['pps_actual_20']:
                        df.at[idx, col] = np.clip(df.at[idx, col] * (1 + (np.random.choice([-1, 1]) * 0.6)), 0, 20)
                    else:
                        df.at[idx, col] = max(0, df.at[idx, col] * factor)

    # ---------------------- Etiqueta de deserción con ruido de etiqueta ----------------------
    if label_noise is not None and 0 < label_noise < 1:
        n_flip = int(round(label_noise * n))
        flip_idx = np.random.choice(df.index, size=n_flip, replace=False)
        df.loc[flip_idx, 'desercion'] = 1 - df.loc[flip_idx, 'desercion']

    # ---------------------- Variables categóricas numéricas para entrenamiento directo ----------------------
    # Codificar tipo_financiamiento a números (orden no implica jerarquía)
    tipo_map = {v: i for i, v in enumerate(sorted(df['tipo_financiamiento'].unique()))}
    df['tipo_financiamiento_code'] = df['tipo_financiamiento'].map(tipo_map).astype(int)

    ingresos_map = {'Muy bajo': 0, 'Bajo': 1, 'Medio': 2, 'Alto': 3}
    if 'ingresos_mensual_cat' in df.columns:
        df['ingresos_mensual_cat_code'] = df['ingresos_mensual_cat'].map(ingresos_map).fillna(0).astype(int)

    # deuda binaria
    df['deuda_academica_flag'] = (df['debt_indicator'] > 0).astype(int)

    # Asegurar tipos limpios y sin NaNs
    df = df.fillna(0)


    # Añadir algunas columnas categóricas derivadas que aparecen en el diccionario
    # ingresos_mensual_cat por cuartiles (sustituye SM cuando no hay valor local)
    q = np.quantile(ingresos, [0.25, 0.5, 0.75])
    def ingresos_cat(x):
        if x <= q[0]:
            return 'Muy bajo'
        elif x <= q[1]:
            return 'Bajo'
        elif x <= q[2]:
            return 'Medio'
        else:
            return 'Alto'

    df['ingresos_mensual_cat'] = [ingresos_cat(x) for x in df['ingresos_mensual']]

    # deuda_academica (binaria y categórica)
    df['deuda_academica'] = np.where(df['debt_indicator'] == 1, 'Con deuda', 'Sin deuda')

    # costo_servicio_mensual: proporcional al indice financiero inverso (simulado)
    # aquí lo generamos como porcentaje del ingreso
    df['costo_servicio_mensual'] = np.round(np.clip(df['ingresos_mensual'] * np.random.uniform(0.05, 0.3, size=n), 10, None), 0)

    # tipo_financiamiento en formato legible (ya generado)

    # Exportar CSV
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), 'simulated_students_1000.csv')
    df.to_csv(out_path, index=False)
    return df, out_path


if __name__ == '__main__':
    df, path = generate(n=1000, seed=42)
    print(f"Dataset simulado generado: {path}")
    print(df.describe(include='all').transpose().head(20))
