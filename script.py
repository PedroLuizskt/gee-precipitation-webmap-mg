# %% [markdown]
"""
╔══════════════════════════════════════════════════════════════════════╗
║   🌧  WEBMAP — PRECIPITAÇÃO FEVEREIRO | ZONA DA MATA MG            ║
║   Fonte: CHIRPS V3 SAT (UCSB-CHC) via Google Earth Engine          ║
║   Cobertura: 2006–2026 | Municípios: Zona da Mata e Sul MG          ║
║   Asset: UCSB-CHC/CHIRPS/V3/DAILY_SAT (IMERG-based, v3.0)         ║
╚══════════════════════════════════════════════════════════════════════╝

DECISÃO DO ASSET:
  ✅ DAILY_SAT  — IMERG-based,real-time, melhor para eventos
                  convectivos extremos como Fev/2026. Cobre até
                  2026-02-28. Escolha ideal para este estudo.
  ℹ  DAILY_RNL  — ERA5-based, climatologicamente mais consistente,
                  mas menos ágil para eventos recentes.
"""

# %% ── CÉLULA 1: IMPORTAÇÕES E CONFIGURAÇÃO ──────────────────────────────────

import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import os
import unicodedata
import re
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Municípios alvo ──────────────────────────────────────────────────────────
MUNICIPIOS_ALVO = [
    "Juiz de Fora",        # Principal afetado — decreto de calamidade
    "Ubá",                 # Colapso de infraestrutura
    "Senador Firmino",
    "Barão de Monte Alto",
    "Caparaó",
    "Divinésia",
    "Dores do Turvo",
    "Durandé",
    "Leopoldina",
    "Matipó",
    "Muriaé",
    "Patrocínio do Muriaé",
    "Paula Cândido",
    "Pequeri",
    "Viçosa",              # Alto volume de precipitação
    "Orizânia",            # Rio Carangola transbordou
    "São João del-Rei",    # Transbordamento do Córrego do Lenheiro
]

# ─── Caminhos ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PATH_MUNICIPIOS = BASE_DIR / "data" / "BR_Municipios_2024" / "BR_Municipios_2024.shp"
PATH_OUTPUT = BASE_DIR / "output"
PATH_CACHE_CSV   = PATH_OUTPUT / "cache_precip_fevereiro_chirps_v3.csv"
PATH_WEBMAP_HTML = PATH_OUTPUT / "webmap_precipitacao_fevereiro.html"
PATH_OUTPUT.mkdir(parents=True, exist_ok=True)

# ─── Parâmetros GEE ───────────────────────────────────────────────────────────
ASSET_ID     = "UCSB-CHC/CHIRPS/V3/DAILY_SAT"   # Near-real-time, IMERG-based
SCALE_METERS = 5566
START_YEAR   = 2006
END_YEAR     = 2026
CRS_GEO      = "EPSG:4674"

print("✅ Configuração carregada.")
print(f"   Asset  : {ASSET_ID}")
print(f"   Período: {START_YEAR}–{END_YEAR} (fevereiro)")
print(f"   Municípios: {len(MUNICIPIOS_ALVO)}")
print(f"   Output : {PATH_OUTPUT}")


# %% ── CÉLULA 2: AUTENTICAÇÃO GOOGLE EARTH ENGINE ─────────────────────────────

try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print("✅ GEE inicializado (high-volume endpoint).")
except Exception:
    ee.Authenticate()
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print("✅ GEE autenticado e inicializado.")


# %% ── CÉLULA 3: CARREGAR E FILTRAR MUNICÍPIOS ────────────────────────────────

def normalize(text: str) -> str:
    """Remove acentos e normaliza string para comparação."""
    if not isinstance(text, str):
        return ""
    return (unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8')
            .lower()
            .strip())

print("📂 Carregando shapefile de municípios...")
gdf_all   = gpd.read_file(PATH_MUNICIPIOS)
gdf_mg    = gdf_all[gdf_all['SIGLA_UF'] == 'MG'].copy()
gdf_mg['norm'] = gdf_mg['NM_MUN'].apply(normalize)

# Mapear nomes normalizados → nome original do alvo
norms_alvo = {normalize(m): m for m in MUNICIPIOS_ALVO}

gdf_alvo = gdf_mg[gdf_mg['norm'].isin(norms_alvo.keys())].copy()
gdf_alvo = gdf_alvo.dissolve(by='NM_MUN').reset_index()
gdf_alvo = gdf_alvo[['NM_MUN', 'geometry']].copy()
gdf_alvo = gdf_alvo.to_crs(CRS_GEO)
gdf_alvo['norm'] = gdf_alvo['NM_MUN'].apply(normalize)

print(f"✅ {len(gdf_alvo)} / {len(MUNICIPIOS_ALVO)} municípios encontrados:")
for nm in sorted(gdf_alvo['NM_MUN'].tolist()):
    print(f"   • {nm}")

nao_encontrados = [m for m in MUNICIPIOS_ALVO
                   if normalize(m) not in gdf_alvo['norm'].values]
if nao_encontrados:
    print(f"\n⚠️  Não encontrados (verificar grafia no shapefile):")
    for m in nao_encontrados:
        print(f"   ✗ {m}")


# %% ── CÉLULA 4: EXTRAÇÃO DE PRECIPITAÇÃO DE FEVEREIRO (COM CACHE) ───────────

def extract_feb_precip_year(year: int, gdf: gpd.GeoDataFrame) -> dict:
    """
    Extrai a precipitação acumulada de fevereiro para cada município via GEE.

    Args:
        year: Ano a processar.
        gdf:  GeoDataFrame com os municípios (CRS geográfico).

    Returns:
        Dict {NM_MUN: total_mm}.
    """
    chirps = ee.ImageCollection(ASSET_ID)

    start = ee.Date.fromYMD(year, 2, 1)
    end   = ee.Date.fromYMD(year, 3, 1)   # início de março, exclusivo

    feb_total = (chirps
                 .filterDate(start, end)
                 .select('precipitation')
                 .sum())

    results = {}
    for _, row in gdf.iterrows():
        name    = row['NM_MUN']
        geom_ee = ee.Geometry(row.geometry.__geo_interface__)
        try:
            val = (feb_total
                   .reduceRegion(
                       reducer    = ee.Reducer.mean(),
                       geometry   = geom_ee,
                       scale      = SCALE_METERS,
                       maxPixels  = 1e10,
                       bestEffort = True,
                   )
                   .get('precipitation')
                   .getInfo())
            results[name] = round(float(val), 2) if val is not None else np.nan
        except Exception as exc:
            print(f"    ⚠️  {name} ({year}): {exc}")
            results[name] = np.nan
    return results


# ─── Verificar cache ──────────────────────────────────────────────────────────
USE_CACHE = False
if PATH_CACHE_CSV.exists():
    try:
        df_cached   = pd.read_csv(PATH_CACHE_CSV)
        anos_cache  = set(df_cached['Ano'].unique())
        muns_cache  = set(df_cached['Municipio'].unique())
        anos_alvo   = set(range(START_YEAR, END_YEAR + 1))
        muns_alvo   = set(gdf_alvo['NM_MUN'].tolist())
        if anos_alvo <= anos_cache and muns_alvo <= muns_cache:
            df_main   = df_cached
            USE_CACHE = True
            print(f"📦 Cache válido encontrado: {PATH_CACHE_CSV.name}")
            print(f"   {len(df_main)} registros | Anos: {sorted(anos_cache)}")
    except Exception as e:
        print(f"⚠️  Erro ao ler cache: {e}. Reprocessando via GEE...")

if not USE_CACHE:
    anos_faltantes = set(range(START_YEAR, END_YEAR + 1))
    if PATH_CACHE_CSV.exists():
        try:
            df_cached      = pd.read_csv(PATH_CACHE_CSV)
            anos_faltantes = set(range(START_YEAR, END_YEAR + 1)) - set(df_cached['Ano'].unique())
        except Exception:
            df_cached = pd.DataFrame(columns=['Ano', 'Municipio', 'Precip_Feb_mm'])
    else:
        df_cached = pd.DataFrame(columns=['Ano', 'Municipio', 'Precip_Feb_mm'])

    print(f"\n🌍 Extraindo CHIRPS V3 SAT para Fevereiro ({START_YEAR}–{END_YEAR})...")
    print(f"   Anos a processar: {sorted(anos_faltantes)}")
    print("   Estimativa: ~30–90 segundos por ano\n")

    records = []
    for year in sorted(anos_faltantes):
        print(f"  → {year}  ", end="", flush=True)
        try:
            data = extract_feb_precip_year(year, gdf_alvo)
            for mun, val in data.items():
                records.append({'Ano': year, 'Municipio': mun, 'Precip_Feb_mm': val})
            vals = [v for v in data.values() if not np.isnan(v)]
            media = np.mean(vals) if vals else 0
            jf    = data.get('Juiz de Fora', 0)
            if year == 2026:
                print(f"✅  JF={jf:.1f}mm | média_reg={media:.1f}mm  🚨")
            else:
                print(f"✅  média_reg={media:.1f}mm")
        except Exception as exc:
            print(f"❌  {exc}")

    df_new  = pd.DataFrame(records)
    df_main = pd.concat([df_cached, df_new], ignore_index=True)
    df_main = df_main.drop_duplicates(subset=['Ano','Municipio'], keep='last')
    df_main.to_csv(PATH_CACHE_CSV, index=False)
    print(f"\n✅ Cache salvo: {PATH_CACHE_CSV}")

print(f"\n📊 Dataset: {len(df_main)} registros | "
      f"{df_main['Ano'].nunique()} anos | "
      f"{df_main['Municipio'].nunique()} municípios")


# %% ── CÉLULA 5: CÁLCULO DE ESTATÍSTICAS ─────────────────────────────────────

anos       = sorted(df_main['Ano'].astype(int).unique().tolist())
municipios = sorted(df_main['Municipio'].unique().tolist())

# Pivot table: índice=Ano, colunas=Município
df_pivot = df_main.pivot_table(index='Ano', columns='Municipio',
                               values='Precip_Feb_mm', aggfunc='mean')

df_hist = df_pivot[df_pivot.index < 2026]

# ─── Estatísticas por município ───────────────────────────────────────────────
stats = {}
for m in municipios:
    if m not in df_pivot.columns:
        continue
    serie_hist = df_hist[m].dropna()
    val_2026   = (df_pivot.loc[2026, m]
                  if 2026 in df_pivot.index and not np.isnan(df_pivot.loc[2026, m])
                  else np.nan)
    media_h    = float(serie_hist.mean())  if len(serie_hist) else np.nan
    std_h      = float(serie_hist.std())   if len(serie_hist) > 1 else np.nan
    max_h      = float(serie_hist.max())   if len(serie_hist) else np.nan
    min_h      = float(serie_hist.min())   if len(serie_hist) else np.nan
    ano_max_h  = int(serie_hist.idxmax())  if len(serie_hist) else 0
    anom_mm    = float(val_2026 - media_h) if not np.isnan(val_2026) else np.nan
    anom_pct   = float(anom_mm / media_h * 100) if media_h else np.nan
    sigma      = float(anom_mm / std_h)    if std_h else np.nan
    percentil  = (float((serie_hist < val_2026).mean() * 100)
                  if not np.isnan(val_2026) else np.nan)

    serie_dict = {}
    for y in anos:
        if y in df_pivot.index and m in df_pivot.columns:
            v = df_pivot.loc[y, m]
            if not np.isnan(v):
                serie_dict[str(y)] = round(float(v), 1)

    stats[m] = {
        'media':       round(media_h, 1) if not np.isnan(media_h) else None,
        'std':         round(std_h,   1) if std_h and not np.isnan(std_h) else None,
        'max_hist':    round(max_h,   1) if not np.isnan(max_h)   else None,
        'min_hist':    round(min_h,   1) if not np.isnan(min_h)   else None,
        'ano_max':     ano_max_h,
        'val_2026':    round(float(val_2026), 1) if not np.isnan(val_2026) else None,
        'anom_mm':     round(anom_mm, 1)  if not np.isnan(anom_mm)  else None,
        'anom_pct':    round(anom_pct, 1) if not np.isnan(anom_pct) else None,
        'sigma':       round(sigma, 2)    if sigma and not np.isnan(sigma)  else None,
        'percentil':   round(percentil,1) if not np.isnan(percentil) else None,
        'serie':       serie_dict,
    }

# ─── Dados por ano para o mapa ────────────────────────────────────────────────
precip_by_year = {}
for y in anos:
    if y in df_pivot.index:
        d = {}
        for m in municipios:
            if m in df_pivot.columns:
                v = df_pivot.loc[y, m]
                if not np.isnan(v):
                    d[m] = round(float(v), 1)
        precip_by_year[str(y)] = d

# ─── Média regional por ano ───────────────────────────────────────────────────
media_regional = {}
for y in anos:
    d = precip_by_year.get(str(y), {})
    vals = [v for v in d.values() if v is not None]
    media_regional[str(y)] = round(np.mean(vals), 1) if vals else 0.0

# ─── Impressão do sumário ─────────────────────────────────────────────────────
print("\n" + "═"*65)
print(f"  SUMÁRIO ESTATÍSTICO — FEVEREIRO 2026 vs HISTÓRICO")
print("═"*65)
print(f"  {'Município':<22} {'Hist.(mm)':<11} {'2026(mm)':<10} {'Anomalia'}")
print(f"  {'─'*62}")
for m in sorted(municipios):
    s = stats.get(m, {})
    med  = f"{s.get('media',0) or 0:.0f}"
    v26  = f"{s.get('val_2026',0) or 0:.0f}"
    ap   = s.get('anom_pct') or 0
    flag = "🚨" if ap > 100 else ("⚠️ " if ap > 50 else "  ")
    print(f"  {m:<22} {med:<11} {v26:<10} +{ap:.0f}% {flag}")

# ─── Preparar GeoJSON dos municípios ─────────────────────────────────────────
gdf_json = gdf_alvo[['NM_MUN', 'geometry']].copy()
geojson_mun = json.loads(gdf_json.to_json())
for feat in geojson_mun['features']:
    feat['properties']['name'] = feat['properties']['NM_MUN']

# Centro do mapa
bounds     = gdf_alvo.total_bounds
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2

print(f"\n✅ Estatísticas calculadas para {len(municipios)} municípios.")
print(f"   Centro do mapa: [{center_lat:.4f}, {center_lon:.4f}]")


# %% ── CÉLULA 6: GERAR WEBMAP HTML ───────────────────────────────────────────

# Serializar dados para JavaScript
PRECIP_JS   = json.dumps(precip_by_year,  ensure_ascii=False)
STATS_JS    = json.dumps(stats,           ensure_ascii=False)
MUNICIPIOS_JS = json.dumps(sorted(municipios), ensure_ascii=False)
ANOS_JS     = json.dumps(sorted(anos),    ensure_ascii=False)
GEOJSON_JS  = json.dumps(geojson_mun,     ensure_ascii=False)
MEDIA_REG_JS = json.dumps(media_regional, ensure_ascii=False)

ANO_OPTIONS = "\n".join(
    f'<option value="{y}" {"selected" if y == 2026 else ""}>'
    f'{y}{"  ─── 🚨 RECORDE HISTÓRICO" if y == 2026 else ""}'
    f'</option>'
    for y in sorted(anos, reverse=True)
)

MUN_OPTIONS = "\n".join(
    f'<option value="{m}">{m}</option>'
    for m in sorted(municipios)
)

DATA_HOJE = datetime.now().strftime('%d/%m/%Y')

# ─────────────────────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Precipitação Fevereiro · Zona da Mata MG</title>

<!-- Leaflet.js 1.9 -->
<link  rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<!-- Chart.js 4 + annotation plugin -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>

<!-- Fonts: Barlow Condensed (impacto) + Karla (corpo) -->
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800&family=Karla:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ══ RESET & BASE ══════════════════════════════════════════════════════════ */
*, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
:root {{
  --bg:          #060d16;
  --bg-card:     #0d1b2a;
  --bg-card2:    #122032;
  --border:      #1a3347;
  --border2:     #254460;
  --blue:        #00b4d8;
  --blue-dim:    #0077b6;
  --red:         #ef233c;
  --red-dim:     #9b1a27;
  --amber:       #f4a261;
  --green:       #52b788;
  --txt:         #caf0f8;
  --txt-dim:     #6e8fa8;
  --txt-muted:   #3d5a72;
  --font-head:   'Barlow Condensed', sans-serif;
  --font-body:   'Karla', sans-serif;
}}
body {{
  font-family: var(--font-body);
  background: var(--bg);
  color: var(--txt);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}

/* ══ HEADER ════════════════════════════════════════════════════════════════ */
#header {{
  background: linear-gradient(90deg, #040b14 0%, #0d1b2a 60%, #0a1520 100%);
  border-bottom: 1px solid var(--border);
  padding: 0 20px;
  height: 52px;
  display: flex;
  align-items: center;
  gap: 18px;
  flex-shrink: 0;
  box-shadow: 0 2px 16px rgba(0,0,0,.5);
  position: relative;
  z-index: 500;
}}
.hdr-icon {{ font-size: 22px; }}
.hdr-text h1 {{
  font-family: var(--font-head);
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: #e0f4ff;
  line-height: 1.1;
}}
.hdr-text p {{
  font-size: 10px;
  color: var(--txt-dim);
  letter-spacing: 0.3px;
  margin-top: 1px;
}}
.badge-alert {{
  margin-left: auto;
  background: var(--red);
  color: #fff;
  padding: 4px 12px;
  border-radius: 20px;
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  animation: blink 2.5s ease-in-out infinite;
  box-shadow: 0 0 12px rgba(239,35,60,.4);
}}
@keyframes blink {{ 0%,100%{{opacity:1;box-shadow:0 0 12px rgba(239,35,60,.4)}} 50%{{opacity:.7;box-shadow:0 0 20px rgba(239,35,60,.7)}} }}

/* ══ LAYOUT ════════════════════════════════════════════════════════════════ */
#main {{ display:flex; flex:1; overflow:hidden; }}
#map  {{ flex:1; position:relative; }}

/* ══ SIDE PANEL ════════════════════════════════════════════════════════════ */
#panel {{
  width: 420px;
  background: var(--bg-card);
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: -6px 0 24px rgba(0,0,0,.4);
}}

/* ══ TABS ══════════════════════════════════════════════════════════════════ */
#tabs {{
  display: flex;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}}
.tab {{
  flex: 1;
  padding: 11px 6px;
  text-align: center;
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  color: var(--txt-dim);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all .2s;
}}
.tab.active {{
  color: var(--blue);
  border-bottom-color: var(--blue);
  background: rgba(0,180,216,.04);
}}
.tab:hover:not(.active) {{ color: #a0cfe8; }}

/* ══ CONTENT AREA ══════════════════════════════════════════════════════════ */
.tab-content {{
  flex: 1;
  overflow-y: auto;
  display: none;
  flex-direction: column;
  gap: 10px;
  padding: 14px;
}}
.tab-content.active {{ display: flex; }}

/* Scrollbar */
.tab-content::-webkit-scrollbar {{ width: 3px; }}
.tab-content::-webkit-scrollbar-track {{ background: var(--bg); }}
.tab-content::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius:2px; }}

/* ══ CARDS ═════════════════════════════════════════════════════════════════ */
.card {{
  background: var(--bg-card2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  flex-shrink: 0;
}}
.card.alert-card {{ border-color: var(--red-dim); }}
.card h4 {{
  font-family: var(--font-head);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  color: var(--txt-dim);
  margin-bottom: 10px;
}}

/* ══ YEAR SELECTOR ═════════════════════════════════════════════════════════ */
#year-select {{
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border2);
  color: var(--txt);
  padding: 9px 12px;
  border-radius: 7px;
  font-family: var(--font-head);
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  outline: none;
  transition: border-color .2s;
}}
#year-select:focus {{ border-color: var(--blue); }}

/* ══ NARRATIVE ═════════════════════════════════════════════════════════════ */
#narrative {{
  padding: 11px 13px;
  border-left: 3px solid var(--blue);
  border-radius: 0 8px 8px 0;
  background: rgba(0,180,216,.06);
  font-size: 12px;
  line-height: 1.65;
  color: #a0d4ee;
  transition: all .3s;
}}
#narrative.is-alert {{
  border-left-color: var(--red);
  background: rgba(239,35,60,.07);
  color: #ffb3bc;
}}

/* ══ KPI GRID ══════════════════════════════════════════════════════════════ */
.kpi-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}}
.kpi {{
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  text-align: center;
}}
.kpi-val {{
  font-family: var(--font-head);
  font-size: 22px;
  font-weight: 800;
  color: var(--blue);
  letter-spacing: 0.5px;
  line-height: 1;
}}
.kpi-val.red    {{ color: var(--red);   }}
.kpi-val.green  {{ color: var(--green); }}
.kpi-val.amber  {{ color: var(--amber); }}
.kpi-lbl {{
  font-size: 9px;
  color: var(--txt-dim);
  margin-top: 4px;
  letter-spacing: 0.3px;
}}

/* ══ RANKING ═══════════════════════════════════════════════════════════════ */
.rank-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 8px;
  border-radius: 6px;
  margin-bottom: 3px;
  background: var(--bg);
  border: 1px solid var(--border);
  cursor: pointer;
  transition: border-color .15s, background .15s;
}}
.rank-item:hover {{ border-color: var(--blue); background: rgba(0,180,216,.04); }}
.rank-item.selected {{ border-color: #fff !important; background: rgba(255,255,255,.04) !important; }}
.rank-num {{
  font-family: var(--font-head);
  font-weight: 700;
  color: var(--txt-muted);
  width: 18px;
  text-align: center;
  font-size: 11px;
}}
.rank-name {{ flex:1; font-size:12px; color: var(--txt); }}
.rank-val  {{
  font-family: var(--font-head);
  font-size: 14px;
  font-weight: 700;
  color: var(--blue);
}}
.rank-bar {{ height:3px; background:var(--border); border-radius:2px; margin-top:3px; }}
.rank-fill {{ height:3px; background:var(--blue-dim); border-radius:2px; transition:width .6s ease; }}
.rank-item.rank-alert .rank-val {{ color: var(--red); }}
.rank-item.rank-alert .rank-fill {{ background: var(--red-dim); }}

/* ══ MUN DETAIL ════════════════════════════════════════════════════════════ */
.stat-row {{
  display:flex; justify-content:space-between; align-items:center;
  padding: 5px 0;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
}}
.stat-row:last-child {{ border-bottom:none; }}
.stat-lbl {{ color: var(--txt-dim); }}
.stat-val {{ font-weight:600; color: #c8e8ff; }}
.stat-val.red {{ color: var(--red); }}
.stat-val.green {{ color: var(--green); }}

/* ══ CHARTS ════════════════════════════════════════════════════════════════ */
.chart-wrap {{ position:relative; }}
.chart-wrap canvas {{ max-width:100%; }}
.h-180 {{ height:180px; }}
.h-220 {{ height:220px; }}
.h-260 {{ height:260px; }}

/* ══ HINT ══════════════════════════════════════════════════════════════════ */
.hint {{
  font-size:11px; color:var(--txt-muted); text-align:center;
  padding:14px 0; font-style:italic;
}}

/* ══ SUMMARY TABLE ═════════════════════════════════════════════════════════ */
.tbl-head, .tbl-row {{
  display: grid;
  grid-template-columns: 1.8fr 1fr 1fr 1fr;
  gap: 4px;
  padding: 5px 4px;
  font-size: 10px;
  align-items: center;
}}
.tbl-head {{
  font-family: var(--font-head);
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--txt-dim);
  border-bottom: 2px solid var(--border2);
  padding-bottom: 6px;
  margin-bottom: 2px;
}}
.tbl-row {{ border-bottom: 1px solid var(--border); }}
.tbl-row:hover {{ background: rgba(0,180,216,.04); }}
.tbl-nm  {{ color: var(--txt); font-weight:500; }}
.tbl-med {{ color: var(--txt-dim); text-align:right; }}
.tbl-26  {{ color: var(--red); font-weight:700; text-align:right; }}
.tbl-an  {{ font-weight:700; text-align:right; }}

/* ══ LEGEND ════════════════════════════════════════════════════════════════ */
#legend {{
  position: absolute;
  bottom: 28px;
  right: 428px;
  z-index: 800;
  background: rgba(6,13,22,.92);
  border: 1px solid var(--border2);
  border-radius: 10px;
  padding: 12px 15px;
  min-width: 160px;
  backdrop-filter: blur(6px);
  box-shadow: 0 4px 16px rgba(0,0,0,.4);
}}
#legend h5 {{
  font-family: var(--font-head);
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  color: var(--txt-dim);
  margin-bottom: 7px;
}}
.leg-bar {{
  height: 10px;
  border-radius: 5px;
  background: linear-gradient(to right, #deebf7, #9ecae1, #3182bd, #08306b);
  margin-bottom: 4px;
}}
.leg-labels {{
  display:flex; justify-content:space-between;
  font-size: 10px; color: var(--txt-dim);
}}
.leg-2026 {{
  margin-top: 7px;
  font-size: 10px;
  color: var(--red);
  font-weight: 600;
  padding-top: 6px;
  border-top: 1px solid var(--border);
}}

/* ══ MAP TOOLTIP ═══════════════════════════════════════════════════════════ */
.lf-tip {{
  background: #07111d !important;
  border: 1px solid var(--border2) !important;
  color: var(--txt) !important;
  font-family: var(--font-body) !important;
  font-size: 12px !important;
  padding: 7px 11px !important;
  border-radius: 7px !important;
  box-shadow: 0 3px 10px rgba(0,0,0,.5) !important;
}}

/* ══ LOADING ════════════════════════════════════════════════════════════════ */
#loading {{
  position:fixed; inset:0; background:var(--bg);
  display:flex; align-items:center; justify-content:center;
  z-index:9999; flex-direction:column; gap:14px;
}}
.spinner {{
  width:42px; height:42px;
  border: 3px solid var(--border);
  border-top-color: var(--blue);
  border-radius:50%;
  animation: spin .7s linear infinite;
}}
@keyframes spin {{ to{{transform:rotate(360deg)}} }}
#loading p {{ font-size:13px; color:var(--txt-dim); letter-spacing:0.3px; }}
</style>
</head>
<body>

<!-- Loading screen -->
<div id="loading">
  <div class="spinner"></div>
  <p>Carregando dados pluviométricos...</p>
</div>

<!-- Header -->
<div id="header">
  <div class="hdr-icon">🌧</div>
  <div class="hdr-text">
    <h1>Precipitação Fevereiro — Zona da Mata &amp; Sul de MG</h1>
    <p>CHIRPS V3 SAT · {START_YEAR}–{END_YEAR} · {len(municipios)} municípios · Fonte: UCSB/CHC via Google Earth Engine · {DATA_HOJE}</p>
  </div>
  <div class="badge-alert">🚨 2026 RECORDE</div>
</div>

<!-- Main container -->
<div id="main">
  <!-- Map -->
  <div id="map"></div>

  <!-- Side panel -->
  <div id="panel">
    <div id="tabs">
      <div class="tab active" onclick="switchTab('year')">📅 ANO ESPECÍFICO</div>
      <div class="tab" onclick="switchTab('hist')">📈 SÉRIE 20 ANOS</div>
    </div>

    <!-- ═══ TAB: ANO ESPECÍFICO ═════════════════════════════════════════════ -->
    <div id="tab-year" class="tab-content active">

      <div class="card">
        <h4>Selecionar Ano</h4>
        <select id="year-select" onchange="updateYear(this.value)">
          {ANO_OPTIONS}
        </select>
      </div>

      <div id="narrative" class="is-alert">Carregando narrativa...</div>

      <div class="card">
        <h4>KPIs — Fevereiro <span id="kpi-year-lbl">2026</span></h4>
        <div class="kpi-grid" id="kpi-grid"></div>
      </div>

      <div class="card">
        <h4>Ranking — Precipitação Acumulada · <span id="rank-year-lbl">2026</span></h4>
        <div id="ranking"></div>
      </div>

      <div class="card" id="mun-detail-wrap" style="display:none">
        <h4>Município Selecionado: <span id="mun-detail-name" style="color:var(--blue)">—</span></h4>
        <div id="mun-stats"></div>
        <div class="chart-wrap h-180" style="margin-top:12px">
          <canvas id="mun-series-chart"></canvas>
        </div>
      </div>

    </div>

    <!-- ═══ TAB: SÉRIE HISTÓRICA ════════════════════════════════════════════ -->
    <div id="tab-hist" class="tab-content">

      <div class="card alert-card">
        <h4 style="color:var(--red)">🚨 Contexto Histórico — Fevereiro 2026</h4>
        <p style="font-size:11px;line-height:1.6;color:#f0a0b0">
          Juiz de Fora acumulou <strong>mais de 579 mm até o dia 24</strong>, superando em
          <strong>3×</strong> a média histórica de ~170 mm. O evento tornou-se o
          fevereiro mais chuvoso registrado na cidade, causando mais de
          <strong>60 mortes</strong> e levando dezenas de municípios a decretar
          estado de calamidade. O gráfico regional abaixo exibe 2026 em destaque.
        </p>
      </div>

      <div class="card">
        <h4>Precipitação Média Regional — Fevereiro (todos os municípios)</h4>
        <div class="chart-wrap h-220">
          <canvas id="regional-chart"></canvas>
        </div>
      </div>

      <div class="card">
        <h4>Série Individual por Município</h4>
        <select id="mun-hist-select" style="width:100%;background:var(--bg);border:1px solid var(--border2);color:var(--txt);padding:8px 12px;border-radius:7px;font-family:var(--font-body);font-size:13px;outline:none;cursor:pointer;margin-bottom:12px" onchange="buildMunHistChart(this.value)">
          {MUN_OPTIONS}
        </select>
        <div class="chart-wrap h-220">
          <canvas id="mun-hist-chart"></canvas>
        </div>
      </div>

      <div class="card">
        <h4>Tabela — 2026 vs Média Histórica</h4>
        <div id="summary-table"></div>
      </div>

    </div>
  </div>
</div>

<!-- Map legend -->
<div id="legend">
  <h5>Precipitação (mm)</h5>
  <div class="leg-bar"></div>
  <div class="leg-labels">
    <span id="leg-min">0 mm</span>
    <span id="leg-max">—</span>
  </div>
  <div class="leg-2026" id="leg-note" style="display:none">⚠️ Escala ajustada para 2026</div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════════════
     JAVASCRIPT
═══════════════════════════════════════════════════════════════════════════ -->
<script>
/* ── DADOS EMBUTIDOS ─────────────────────────────────────────────────────── */
const PRECIP_BY_YEAR = {PRECIP_JS};
const STATS_MUN      = {STATS_JS};
const MUNICIPIOS     = {MUNICIPIOS_JS};
const ANOS           = {ANOS_JS};
const GEOJSON        = {GEOJSON_JS};
const MEDIA_REG      = {MEDIA_REG_JS};

/* ── NARRATIVAS ──────────────────────────────────────────────────────────── */
const NARRATIVES = {{
  "2026": `🚨 <strong>Fevereiro de 2026 — Catástrofe Histórica</strong><br>
    Chuvas sem precedentes devastaram a Zona da Mata Mineira. Juiz de Fora
    acumulou <strong>+579 mm até o dia 24</strong>, superando 3× a média histórica
    de ~170 mm. Mais de <strong>60 mortes</strong>, milhares de desalojados e
    severos danos à infraestrutura. Dezenas de municípios em calamidade.`,
  "2022": `⚠️ <strong>Fevereiro de 2022</strong><br>
    Precipitações acima da média em Minas Gerais e Rio de Janeiro. Eventos
    extremos registrados em diversas localidades do Sudeste brasileiro.`,
  "2020": `🌧 <strong>Fevereiro de 2020</strong><br>
    Janeiro e fevereiro de 2020 foram marcados por fortes chuvas em MG.
    Precipitações acima da média na maioria dos municípios da região.`,
  "2013": `🌧 <strong>Fevereiro de 2013</strong><br>
    Período com La Niña fraco influenciando o padrão de chuvas no Sudeste.
    Precipitações próximas à média histórica na maior parte da região.`,
  "2009": `🌧 <strong>Fevereiro de 2009</strong><br>
    Ano El Niño moderado, associado a períodos mais chuvosos no Sul e Sudeste
    brasileiro durante o verão.`,
}};
const defaultNarr = (y) => `📊 <strong>Fevereiro de ${{y}}</strong><br>
  Precipitação acumulada no mês de fevereiro de ${{y}} para os municípios da
  Zona da Mata e Sul de MG. Clique em um município no mapa para explorar
  sua série histórica individual e estatísticas.`;

/* ── ESTADO ──────────────────────────────────────────────────────────────── */
let currentYear    = "2026";
let selectedMun    = null;
let geoLayer       = null;
let map;
let chartMunSeries = null;
let chartRegional  = null;
let chartMunHist   = null;

/* ── PALETA DE CORES (Blues) ─────────────────────────────────────────────── */
const STOPS = [
  [0.00, [222,235,247]],
  [0.20, [189,215,231]],
  [0.40, [107,174,214]],
  [0.60, [ 49,130,189]],
  [0.80, [ 23, 78,143]],
  [1.00, [  8, 48,107]],
];

function interpolateColor(t) {{
  t = Math.max(0, Math.min(1, t));
  let lo = STOPS[0], hi = STOPS[STOPS.length-1];
  for (let i=0; i<STOPS.length-1; i++) {{
    if (t >= STOPS[i][0] && t <= STOPS[i+1][0]) {{ lo=STOPS[i]; hi=STOPS[i+1]; break; }}
  }}
  const f = lo[0]===hi[0] ? 0 : (t-lo[0])/(hi[0]-lo[0]);
  return STOPS.map((_,i) => Math.round(lo[1][i] + f*(hi[1][i]-lo[1][i])));
}}

function getColor(val, minV, maxV) {{
  if (val==null||isNaN(val)) return '#1a2a3a';
  const [r,g,b] = interpolateColor((val-minV)/(maxV-minV||1));
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function yearRange(year) {{
  const d = PRECIP_BY_YEAR[String(year)] || {{}};
  const v = Object.values(d).filter(x=>!isNaN(x));
  return v.length ? {{min:Math.min(...v), max:Math.max(...v)}} : {{min:0, max:1}};
}}

/* ══ MAPA ════════════════════════════════════════════════════════════════════ */
function initMap() {{
  map = L.map('map', {{
    center: [{center_lat:.4f},{center_lon:.4f}],
    zoom: 8,
    zoomControl: true,
    preferCanvas: true,
  }});

  const dark = L.tileLayer(
    'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{attribution:'© OpenStreetMap | © CartoDB', opacity:0.85}}
  );
  const sat = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{attribution:'© Esri', opacity:0.7}}
  );
  const osm = L.tileLayer(
    'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{attribution:'© OpenStreetMap', opacity:0.4}}
  );
  dark.addTo(map);
  L.control.layers({{"🌑 Dark":"dark","🛰️ Satélite":"sat","🗺️ OSM":"osm"}}.constructor===Object?
    {{"🌑 Dark":dark,"🛰️ Satélite":sat,"🗺️ OSM":osm}}:{{}},
    {{}},{{position:'topright'}}
  ).addTo(map);

  // Remove Leaflet attribution
  map.attributionControl.setPrefix('');

  buildGeoLayer();
  map.fitBounds(geoLayer.getBounds().pad(0.08));
}}

function buildGeoLayer() {{
  const rng = yearRange(currentYear);
  geoLayer = L.geoJSON(GEOJSON, {{
    style: f => styleFeature(f, currentYear, rng),
    onEachFeature,
  }}).addTo(map);
}}

function styleFeature(feature, year, rng) {{
  const name = feature.properties.name || feature.properties.NM_MUN;
  const data = PRECIP_BY_YEAR[String(year)] || {{}};
  const val  = data[name];
  const sel  = (selectedMun === name);
  return {{
    fillColor:   getColor(val, rng.min, rng.max),
    fillOpacity: 0.80,
    color:       sel ? '#ffffff' : (year==="2026" ? '#ef233c88' : '#00b4d855'),
    weight:      sel ? 3 : 1.2,
    opacity:     0.9,
  }};
}}

function onEachFeature(feature, layer) {{
  const name = feature.properties.name || feature.properties.NM_MUN;
  layer.on('mouseover', function(e) {{
    if (selectedMun !== name)
      layer.setStyle({{color:'#fff', weight:2.5, fillOpacity:0.95}});
    const d   = PRECIP_BY_YEAR[currentYear] || {{}};
    const val = d[name];
    const st  = STATS_MUN[name] || {{}};
    const tip = `<div style="min-width:180px">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:15px;font-weight:700;color:#e0f4ff;margin-bottom:4px">${{name}}</div>
      <div style="font-size:12px;color:#90caf9">Fev ${{currentYear}}: <strong style="color:#fff">${{val!=null?val.toFixed(1):'—'}} mm</strong></div>
      <div style="font-size:10px;color:#6e8fa8;margin-top:2px">Média hist.: ${{st.media||'—'}} mm</div>
    </div>`;
    layer.bindTooltip(tip, {{className:'lf-tip', sticky:true}}).openTooltip(e.latlng);
  }});
  layer.on('mouseout', function() {{
    if (selectedMun !== name) refreshLayerStyle(layer, feature, currentYear);
    layer.closeTooltip();
  }});
  layer.on('click', function() {{ selectMun(name); }});
}}

function refreshAllStyles() {{
  const rng = yearRange(currentYear);
  geoLayer.eachLayer(l => {{
    const name = l.feature.properties.name || l.feature.properties.NM_MUN;
    if (selectedMun === name) return;
    l.setStyle(styleFeature(l.feature, currentYear, rng));
  }});
}}

function refreshLayerStyle(layer, feature, year) {{
  const rng = yearRange(year);
  layer.setStyle(styleFeature(feature, year, rng));
}}

/* ══ SELECIONAR MUNICÍPIO ════════════════════════════════════════════════════ */
function selectMun(name) {{
  selectedMun = name;
  const rng = yearRange(currentYear);
  geoLayer.eachLayer(l => {{
    const n = l.feature.properties.name || l.feature.properties.NM_MUN;
    if (n===name) l.setStyle({{color:'#ffffff',weight:3,fillOpacity:0.92}});
    else l.setStyle(styleFeature(l.feature, currentYear, rng));
  }});

  // Mark ranking item
  document.querySelectorAll('.rank-item').forEach(el => {{
    el.classList.toggle('selected', el.dataset.mun === name);
  }});

  updateMunDetail(name);
  document.getElementById('mun-detail-wrap').style.display = 'block';
  document.getElementById('mun-detail-wrap').scrollIntoView({{behavior:'smooth',block:'nearest'}});
}}

/* ══ ATUALIZAR ANO ═══════════════════════════════════════════════════════════ */
function updateYear(year) {{
  currentYear = String(year);
  document.getElementById('kpi-year-lbl').textContent = year;
  document.getElementById('rank-year-lbl').textContent = year;

  // Map
  refreshAllStyles();

  // Narrative
  const narr = NARRATIVES[year] || defaultNarr(year);
  const box  = document.getElementById('narrative');
  box.innerHTML = narr;
  box.className = year==="2026" ? 'is-alert' : '';

  // Legend
  const rng = yearRange(year);
  document.getElementById('leg-min').textContent = rng.min.toFixed(0)+' mm';
  document.getElementById('leg-max').textContent = rng.max.toFixed(0)+' mm';
  document.getElementById('leg-note').style.display = year==="2026" ? 'block' : 'none';

  // Panels
  updateKPIs(year);
  updateRanking(year);
  if (selectedMun) updateMunDetail(selectedMun);
}}

/* ══ KPIs ════════════════════════════════════════════════════════════════════ */
function updateKPIs(year) {{
  const d    = PRECIP_BY_YEAR[String(year)] || {{}};
  const vals = Object.values(d).filter(v=>!isNaN(v));
  if (!vals.length) return;

  const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
  const mx   = Math.max(...vals);
  const mxM  = Object.entries(d).find(([,v])=>v===mx)?.[0]||'—';

  // Historical mean (excluding 2026)
  let hVals=[];
  ANOS.filter(y=>y<2026).forEach(y=>{{
    const dh = PRECIP_BY_YEAR[String(y)]||{{}};
    Object.values(dh).forEach(v=>{{if(!isNaN(v)) hVals.push(v);}});
  }});
  const hMean  = hVals.length ? hVals.reduce((a,b)=>a+b,0)/hVals.length : mean;
  const anomPct = ((mean-hMean)/hMean*100).toFixed(1);
  const isAlert = year==="2026";

  document.getElementById('kpi-grid').innerHTML = `
    <div class="kpi">
      <div class="kpi-val ${{isAlert?'red':''}}">${{mean.toFixed(1)}}</div>
      <div class="kpi-lbl">Média regional (mm)</div>
    </div>
    <div class="kpi">
      <div class="kpi-val ${{isAlert?'red':''}}">${{mx.toFixed(1)}}</div>
      <div class="kpi-lbl">Máx · ${{mxM}}</div>
    </div>
    <div class="kpi">
      <div class="kpi-val ${{parseFloat(anomPct)>0?(isAlert?'red':'amber'):'green'}}">${{parseFloat(anomPct)>=0?'+':''}}${{anomPct}}%</div>
      <div class="kpi-lbl">Anomalia vs hist.</div>
    </div>
    <div class="kpi">
      <div class="kpi-val">${{vals.length}}</div>
      <div class="kpi-lbl">Municípios</div>
    </div>`;
}}

/* ══ RANKING ═════════════════════════════════════════════════════════════════ */
function updateRanking(year) {{
  const d = PRECIP_BY_YEAR[String(year)] || {{}};
  const sorted = Object.entries(d).filter(([,v])=>!isNaN(v)).sort(([,a],[,b])=>b-a);
  const maxV = sorted.length ? sorted[0][1] : 1;

  document.getElementById('ranking').innerHTML = sorted.map(([name,val],i) => {{
    const pct = ((val/maxV)*100).toFixed(0);
    const isTop = (year==="2026"&&i===0);
    const isSel = (selectedMun===name);
    return `<div class="rank-item ${{isTop?'rank-alert':''}} ${{isSel?'selected':''}}" data-mun="${{name}}" onclick="selectMun('${{name}}')">
      <span class="rank-num">${{i+1}}</span>
      <div style="flex:1">
        <div style="display:flex;justify-content:space-between">
          <span class="rank-name">${{name}}</span>
          <span class="rank-val">${{val.toFixed(1)}} mm</span>
        </div>
        <div class="rank-bar"><div class="rank-fill" style="width:${{pct}}%"></div></div>
      </div>
    </div>`;
  }}).join('');
}}

/* ══ DETALHE MUNICÍPIO ═══════════════════════════════════════════════════════ */
function updateMunDetail(name) {{
  const st   = STATS_MUN[name] || {{}};
  const d    = PRECIP_BY_YEAR[currentYear] || {{}};
  const val  = d[name];
  const anom = st.anom_pct;

  document.getElementById('mun-detail-name').textContent = name;
  document.getElementById('mun-stats').innerHTML = `
    <div class="stat-row">
      <span class="stat-lbl">Fevereiro ${{currentYear}}</span>
      <span class="stat-val ${{currentYear==="2026"?'red':''}}">${{val!=null?val.toFixed(1):'—'}} mm</span>
    </div>
    <div class="stat-row">
      <span class="stat-lbl">Média histórica (2006–2025)</span>
      <span class="stat-val">${{st.media||'—'}} mm</span>
    </div>
    <div class="stat-row">
      <span class="stat-lbl">Máximo histórico</span>
      <span class="stat-val">${{st.max_hist||'—'}} mm (${{st.ano_max||'—'}})</span>
    </div>
    <div class="stat-row">
      <span class="stat-lbl">Mínimo histórico</span>
      <span class="stat-val">${{st.min_hist||'—'}} mm</span>
    </div>
    ${{currentYear==="2026"?`
    <div class="stat-row">
      <span class="stat-lbl">Anomalia 2026 vs média</span>
      <span class="stat-val red">+${{st.anom_mm||0}} mm (+${{(st.anom_pct||0).toFixed(0)}}%)</span>
    </div>
    <div class="stat-row">
      <span class="stat-lbl">Desvios padrão (σ)</span>
      <span class="stat-val red">${{(st.sigma||0).toFixed(1)}}σ</span>
    </div>`:''}}
  `;

  buildMunSeriesChart(name);
}}

/* ══ GRÁFICO SÉRIE MUNICIPAL (tab ano específico) ════════════════════════════ */
function buildMunSeriesChart(name) {{
  const st    = STATS_MUN[name] || {{}};
  const serie = st.serie || {{}};
  const anos  = ANOS.map(String);
  const vals  = anos.map(y => serie[y] ?? 0);
  const media = st.media || 0;

  const bgColors = anos.map(y => y==="2026" ? '#ef233c' : 'rgba(0,180,216,0.55)');
  const bdColors = anos.map(y => y==="2026" ? '#c41631' : 'rgba(0,180,216,0.9)');

  const ctx = document.getElementById('mun-series-chart').getContext('2d');
  if (chartMunSeries) chartMunSeries.destroy();
  chartMunSeries = new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: anos,
      datasets: [{{
        label: 'Precipitação (mm)',
        data:  vals,
        backgroundColor: bgColors,
        borderColor:     bdColors,
        borderWidth: 1,
        borderRadius: 3,
      }}]
    }},
    options: chartOptions({{
      plugins: {{
        legend: {{display:false}},
        annotation: {{ annotations: {{
          media: {{type:'line', yMin:media, yMax:media, borderColor:'#f4a261',
            borderWidth:1.5, borderDash:[4,3],
            label:{{display:true,content:`Média: ${{media.toFixed(0)}} mm`,position:'end',
              color:'#f4a261',font:{{size:8,family:"'Barlow Condensed',sans-serif"}}}}}}
        }}}}
      }},
      scales: {{ x:xScale(), y:yScale() }}
    }})
  }});
}}

/* ══ GRÁFICO REGIONAL (tab histórico) ════════════════════════════════════════ */
function buildRegionalChart() {{
  const anos  = ANOS.map(String);
  const vals  = anos.map(y => MEDIA_REG[y] || 0);
  const histV = vals.filter((_,i) => anos[i]!=="2026");
  const hMean = histV.reduce((a,b)=>a+b,0)/(histV.length||1);

  const bg = anos.map(y => y==="2026"
    ? 'rgba(239,35,60,0.8)'
    : `rgba(0,180,216,0.45)`);
  const bd = anos.map(y => y==="2026" ? '#c41631' : 'rgba(0,180,216,.8)');

  const ctx = document.getElementById('regional-chart').getContext('2d');
  if (chartRegional) chartRegional.destroy();
  chartRegional = new Chart(ctx, {{
    type:'bar',
    data:{{
      labels: anos,
      datasets: [{{
        label: 'Média regional (mm)',
        data:  vals,
        backgroundColor: bg,
        borderColor: bd,
        borderWidth: 1,
        borderRadius: 3,
      }}]
    }},
    options: chartOptions({{
      plugins: {{
        legend:{{display:false}},
        tooltip:{{
          callbacks:{{
            label: c => `${{c.raw.toFixed(1)}} mm`,
            afterLabel: c => c.label==="2026" ? "⚠️ Evento extremo" :
              `Anomalia: ${{((c.raw-hMean)/hMean*100)>=0?'+':''}}${{((c.raw-hMean)/hMean*100).toFixed(1)}}%`,
          }},
          backgroundColor:'#07111d', titleColor:'#e0f4ff',
          bodyColor:'#7fb3d3', borderColor:'#1a3347', borderWidth:1,
        }},
        annotation:{{ annotations:{{
          media: {{type:'line', yMin:hMean, yMax:hMean,
            borderColor:'#f4a261', borderWidth:1.5, borderDash:[5,3],
            label:{{display:true,content:`Hist.: ${{hMean.toFixed(0)}} mm`,position:'end',
              color:'#f4a261',font:{{size:8,family:"'Barlow Condensed',sans-serif"}}}}}}
        }}}}
      }},
      scales:{{ x:xScale(), y:yScale() }}
    }})
  }});
}}

/* ══ GRÁFICO SÉRIE MUNICIPAL (tab histórico) ═════════════════════════════════ */
function buildMunHistChart(name) {{
  const st    = STATS_MUN[name] || {{}};
  const serie = st.serie || {{}};
  const anos  = ANOS.map(String);
  const vals  = anos.map(y => serie[y] ?? null);
  const media = st.media || 0;

  const ptCol = anos.map(y => y==="2026" ? '#ef233c' : '#00b4d8');
  const ptRad = anos.map(y => y==="2026" ? 8 : 3.5);
  const ptHov = anos.map(y => y==="2026" ? 10 : 6);

  const ctx = document.getElementById('mun-hist-chart').getContext('2d');
  if (chartMunHist) chartMunHist.destroy();
  chartMunHist = new Chart(ctx, {{
    type:'line',
    data:{{
      labels: anos,
      datasets: [{{
        label: name,
        data:  vals,
        borderColor: '#00b4d8',
        backgroundColor: 'rgba(0,180,216,0.07)',
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointBackgroundColor: ptCol,
        pointRadius: ptRad,
        pointHoverRadius: ptHov,
        spanGaps: true,
      }}]
    }},
    options: chartOptions({{
      plugins:{{
        legend:{{display:false}},
        tooltip:{{
          callbacks:{{
            label: c => c.raw!=null ? `${{c.raw.toFixed(1)}} mm` : '—',
            afterLabel: c => c.label==="2026" ? "⚠️ Evento extremo" : "",
          }},
          backgroundColor:'#07111d', titleColor:'#e0f4ff',
          bodyColor:'#7fb3d3', borderColor:'#1a3347', borderWidth:1,
        }},
        annotation:{{ annotations:{{
          media:{{type:'line', yMin:media, yMax:media,
            borderColor:'#f4a261', borderWidth:1.5, borderDash:[4,3],
            label:{{display:true,content:`Média: ${{media.toFixed(0)}} mm`,position:'end',
              color:'#f4a261', font:{{size:8,family:"'Barlow Condensed',sans-serif"}}}}}}
        }}}}
      }},
      scales:{{ x:xScale(), y:yScale() }}
    }})
  }});
}}

/* ══ TABELA RESUMO ═══════════════════════════════════════════════════════════ */
function buildSummaryTable() {{
  const rows = MUNICIPIOS.slice().sort().map(m => {{
    const st  = STATS_MUN[m] || {{}};
    const ap  = st.anom_pct || 0;
    const col = ap>150?'#ef5350': ap>80?'#f4a261': ap>30?'#90caf9':'#52b788';
    return `<div class="tbl-row">
      <span class="tbl-nm">${{m}}</span>
      <span class="tbl-med">${{(st.media||0).toFixed(0)}}</span>
      <span class="tbl-26">${{(st.val_2026||0).toFixed(0)}}</span>
      <span class="tbl-an" style="color:${{col}}">+${{ap.toFixed(0)}}%</span>
    </div>`;
  }}).join('');

  document.getElementById('summary-table').innerHTML = `
    <div class="tbl-head">
      <span>Município</span>
      <span style="text-align:right">Média hist.</span>
      <span style="text-align:right">2026</span>
      <span style="text-align:right">Anomalia</span>
    </div>${{rows}}`;
}}

/* ══ UTILITÁRIOS CHART.JS ════════════════════════════════════════════════════ */
function chartOptions(extra) {{
  return Object.assign({{
    responsive: true,
    maintainAspectRatio: false,
    animation: {{duration: 400}},
  }}, extra);
}}
function xScale() {{
  return {{ ticks:{{color:'#6e8fa8',font:{{size:8}},maxRotation:45,minRotation:30}}, grid:{{color:'#1a3347'}} }};
}}
function yScale() {{
  return {{ ticks:{{color:'#6e8fa8',font:{{size:9}}}}, grid:{{color:'#1a3347'}} }};
}}

/* ══ TABS ════════════════════════════════════════════════════════════════════ */
function switchTab(tab) {{
  ['year','hist'].forEach((t,i) => {{
    document.querySelectorAll('.tab')[i].classList.toggle('active', t===tab);
    document.getElementById(`tab-${{t}}`).classList.toggle('active', t===tab);
  }});
  if (tab==='hist') {{
    buildRegionalChart();
    buildMunHistChart(document.getElementById('mun-hist-select').value);
    buildSummaryTable();
  }}
}}

/* ══ INIT ════════════════════════════════════════════════════════════════════ */
window.addEventListener('load', () => {{
  Chart.register(window['chartjs-plugin-annotation']||{{}});
  initMap();
  updateYear("2026");
  setTimeout(() => document.getElementById('loading').style.display='none', 600);
}});
</script>
</body>
</html>"""

# ─── Salvar o arquivo HTML ────────────────────────────────────────────────────
with open(PATH_WEBMAP_HTML, 'w', encoding='utf-8') as f:
    f.write(HTML)

size_kb = PATH_WEBMAP_HTML.stat().st_size / 1024
print(f"\n{'═'*65}")
print(f"  ✅  WEBMAP GERADO COM SUCESSO")
print(f"{'═'*65}")
print(f"  Arquivo : {PATH_WEBMAP_HTML}")
print(f"  Tamanho : {size_kb:.1f} KB")
print(f"  Cache   : {PATH_CACHE_CSV}")
print(f"\n  Para abrir: clique duplo no arquivo .html ou arraste")
print(f"  para o navegador (funciona offline — totalmente autônomo).")
print(f"{'═'*65}")
