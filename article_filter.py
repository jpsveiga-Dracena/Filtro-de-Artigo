import pandas as pd
import numpy as np
import re

# ── 1. Carregar CSV ──────────────────────────────────────────────────────────
df = pd.read_csv('/home/sandbox/lens-export.csv', sep=';', encoding='utf-8-sig', low_memory=False)
df.columns = df.columns.str.strip()
df['Citing Works Count'] = pd.to_numeric(df['Citing Works Count'], errors='coerce').fillna(0).astype(int)
df['Publication Year']   = pd.to_numeric(df['Publication Year'],   errors='coerce').fillna(2000).astype(int)

def txt(row):
    return ' '.join([
        str(row.get('Title','') or ''),
        str(row.get('Abstract','') or ''),
        str(row.get('Keywords','') or ''),
        str(row.get('Fields of Study','') or ''),
    ]).lower()

df['_text'] = df.apply(txt, axis=1)

# ── 2. EXCLUSÕES ABRANGENTES ─────────────────────────────────────────────────
EXCL = [
    # Entomologia/pragas
    r'armyworm', r'spodoptera', r'drosophila', r'tephritid',
    r'honey bee', r'honeybee', r'pollinator.*ecolog',
    r'nematode.*divers', r'meloidogyne',
    r'aphid.*molecul', r'aphid.*morphom', r'sugarcane aphid',
    r'arthropod pest', r'plutella xylostella', r'flea beetle',
    r'rodent pest', r'rodent.*farm',
    r'whitefly.*genetic', r'bemisia tabaci',
    # Animais silvestres / conflito / ecologia animal
    r'leopard', r'sloth bear', r'tiger.*attack', r'elephant.*corridor',
    r'hippopotamus', r'hippo\b', r'carnivore.*attack',
    r'bird.*predation', r'bat.*predation',
    r'red squirrel', r'american.*squirrel',
    r'long.eared owl', r'nesting.*owl', r'owl.*nesting',
    r'migratory bird', r'fruit.*migratory',
    r'jaguar.*corridor', r'corridor.*jaguar',
    r'wildlife.*corridor',
    r'amphibian.*red list', r'red list.*amphibian',
    r'mollusc', r'freshwater snail', r'gastropod',
    r'drymaeus', r'bulimulidae',
    r'acropora', r'coral reef',
    r'donkey.*popul', r'mule.*popul',
    r'sicariid', r'sicarius',
    r'camera trap',
    # Epidemiologia animal/humana
    r'salmonella', r'leishmania', r'tick.borne', r'zoonos',
    r'animal bite', r'rabies', r'brucella', r'anthrax',
    r'seroprevalence', r'seroepidemiolog',
    r'african swine fever', r'swine fever',
    r'newcastle disease',
    r'porcine epidemic diarrhea', r'pedv',
    r'marburg virus', r'ebola', r'diphtheria',
    r'leukemia', r'lymphoid.*leukemia', r'myeloid.*leukemia',
    r'covid', r'coronavirus', r'influenza', r'malaria', r'dengue',
    r'tuberculosis', r'helminth',
    r'virus.bearing pest', r'phytophthora cinnamomi',
    # Ecologia marinha/aquática
    r'sargassum.*coast', r'pelagic.*sargassum',
    r'ship wake', r'shoreline erosion.*ship',
    r'simulation.*marine activit', r'marine activit.*simulation',
    # Ecologia florestal/paisagem sem foco agrícola
    r'atlantic rainforest.*degradation',
    r'deforestation driver', r'context.*deforestation',
    r'scale.*deforestation',
    r'oil palm.*mammal', r'mammal.*oil palm',
    r'tree species.*disturbance gradient',
    r'carbon stock.*plantation forest',
    r'woodland key habitat',
    r'long.eared owl',
    r'aspen.*keystone', r'keystone.*aspen',
    r'juniper.*population', r'population.*juniper',
    r'atlantic forest fragment',
    r'palearctic',
    # Geologia / sismologia / hidrologia não-agrícola
    r'seismic microzonation', r'microzonation',
    r'photo.geolog', r'geological.*3d model',
    r'r\.avaflow', r'avaflow',
    r'morfométrica.*sub.bacia', r'sub.bacia.*morfométrica',
    r'freewat', r'aquifer contamination', r'cregis',
    r'reservoir break flood',
    r'headwaters.*water quality', r'water quality.*headwater',
    # Social / urbano / não-agrícola
    r'violent.*sexual crime', r'sexual crime',
    r'liveability', r'livability',
    r'consumer market.*size', r'consumer.*market.*decision',
    r'mobility pattern.*older', r'social health.*older',
    r'household food insecurity', r'household.*food security',
    r'drought.*migration.*india', r'migration.*drought.*india',
    r'urban green carbon',
    r'no thesis.*future', r'research.*thesis',
    r'necropolit',
    r'periurban.*flood', r'flood.*periurban',
    r'typhoon disaster', r'resilience.*typhoon',
    r'economic loss.*gis.*seism', r'seism.*economic loss',
    # Genômica / QTL / sem GIS
    r'qtl mapping.*germination', r'qtl.*seed vigor',
    r'genome.wide.*diversity.*maize', r'gwas.*maize landrace',
    r'phylogeograph',
    r'microbial diversity.*banana', r'banana.*microb',
    # Outros
    r'telecoupl',
    r'edible insect.*ethnograph',
    r'mass grave',
    r'feral sorghum',
    r'biofuel.*water impact',
    r'oasis.*fachi', r'fachi.*oasis',
    r'evaluation.*cattle farmer.*knowledge',
    r'endemic.*centre.*endem',
    r'ecosystem service.*protected area',
    r'turbine.*energy yield', r'energy yield.*turbine',
    r'catastrophic rock',
    r'anxiolytic drug.*fish',
    r'distribution.*ludwigia',
    r'small hydropower',
    r'threat risk register',
    r'soil viewer.*qgis', r'qgis.*3d soil viewer',
    r'nagoya protocol.*lookup',
    r'old cadastral map', r'cadastral.*georefer',
    r'inland.*maritime.*simulation',
    r'subak',
    r'hotspot.*python.*plugin',
    r'lisam.*liveability',
    r'freewat',
    r'invasion.*biology.*bamboo', r'bamboo.*ecology',
    r'pangasius', r'invasive fish', r'pangasianodon',
    r'ecological corridor.*jaguar',
    r'exploring.*mobility.*social health',
    r'no thesis',
    r'revitaliz.*contaminated soil',
    r'metabolomic.*sorghum',
    r'necropolitical',
    r'scenario.*water spring',
    r'sims.*geospatial matching',
    r'nitrogen.*bacteria.*element',
    r'diphtheria.*outbreak',
    r'grass gis.*simulation.*marine',
    # ── Exclusões adicionais (off-topics confirmados no top 50) ──────────────
    # Item 10: sistemas de produção pecuária (sem foco GIS/agricultura de precisão)
    r'classif.*livestock production system',
    r'livestock production system.*classif',
    r'livestock production system.*characteris',
    # Item 12: arqueologia alimentar / isótopos estáveis / cáries dentárias
    r'dental caries', r'stable isotope.*cari', r'cari.*stable isotope',
    r'isotope.*dental', r'dental.*isotope',
    r'archaeolog.*diet', r'diet.*archaeolog',
    # Item 16: leguminosa multipropósito sem foco GIS
    r'lablab purpureus', r'multi.purpose legume.*lablab', r'lablab.*legume',
    # Item 21: névoa costeira / balanço ecológico sem foco agrícola
    r'coastal fog', r'fog.*ecological balance', r'jizan.*region.*plant',
    # Item 24: pessoas deslocadas internamente / conflito armado
    r'internally displaced person', r'conflict.driven.*displaced',
    r'displaced.*conflict', r'idp.*camp', r'camp.*idp',
    # Item 27: resposta a estresse ambiental / cross-species
    r'cross.species.*environmental stress',
    r'environmental stress.*cross.species',
    r'cross.species.*stress response',
    # Item 37: capital natural + turismo (sem contexto agrícola)
    r'natural capital.*tourism', r'tourism.*natural capital',
    # Item 44: planejamento de rota de drone / NSGA + BWO (não-agrícola)
    r'nsga.*bwo', r'bwo.*nsga',
    r'path planning.*whale.*optim', r'whale.*optim.*path planning',
    r'two.stage hybrid.*path planning', r'path planning.*two.stage hybrid',
    r'multi.objective.*path planning.*uav.*3d',
    # Item 45: peixe invasor / Pangasianodon hipophthalmus
    r'pangasianodon hipophthalmus', r'introduced.*catfish',
    r'invasive catfish', r'catfish.*invasive',
    # Item 49: Musa / banana silvestre-domesticado (genética sem foco GIS)
    r'musa species.*mainland', r'wild.*domesticat.*musa',
    r'domesticat.*musa.*wild', r'banana.*wild.*domesticat',
    r'wild.*banana.*phylogen',
    # Item 50: vacas em tie-stall (bem-estar animal, sem GIS)
    r'tie.stall', r'cow.*tie.stall', r'tie.stall.*cow',
    r'treated.*untreated.*cow', r'untreated.*treated.*cow',
    # Anfíbios / anuros
    r'\banuran\b', r'leptodactyl', r'anura.*conservation',
    # Mark-release-recapture / borboletas de pastagem
    r'mark.release.recapture', r'grassland butterfly', r'butterfly.*micro.habitat',
    # Esquilo-de-chão
    r'spermophilus', r'ground squirrel.*habitat', r'habitat.*ground squirrel',
    # Homogeneização biótica / controle natural de pragas (ecologia pura)
    r'biotic homogenization', r'natural pest control provider',
]

def is_excluded(text):
    for pat in EXCL:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

df['_excl'] = df['_text'].apply(is_excluded)
df_clean = df[~df['_excl']].copy()
print(f"Após exclusão: {len(df_clean)}")

# ── 3. Termos positivos (FOSS4G + GIS operacional aplicado à agricultura) ────
FOSS = [r'qgis', r'qfield', r'foss4g', r'grass gis', r'saga gis', r'geoserver',
        r'postgis', r'openlayers', r'open source gis', r'open-source gis',
        r'free gis', r'orfeo toolbox', r'pyqgis',
        r'free and open source gis', r'free and open-source gis']

UAV_CROP = [r'uav.*crop',r'drone.*crop',r'crop.*uav',r'crop.*drone',
            r'uav.*agri',r'agri.*uav',r'unmanned aerial.*crop',
            r'uav.*precision',r'uav.*yield',r'rpas.*agri',r'aerial.*crop']

SAT_CROP = [r'satellite.*crop',r'crop.*satellite',r'remote sensing.*crop',
            r'crop.*remote sensing',r'multispectral.*crop',r'crop.*multispectral',
            r'hyperspectral.*crop',r'sentinel.*crop',r'landsat.*crop',
            r'modis.*crop',r'imagery.*crop',r'crop.*imagery',r'sar.*crop',
            r'radar.*crop']

PREC_AGRI = [r'precision agriculture',r'precision farming',r'site.specific.*agri',
             r'variable rate.*appl',r'management zone.*agri',
             r'prescription map',r'digital agriculture',r'smart farming',
             r'smart agricultur']

GIS_CROP  = [r'gis.*crop',r'crop.*gis',r'gis.*yield',r'yield.*gis',
             r'gis.*farm\b',r'farm.*gis',r'gis.*agricultur',r'agricultur.*gis',
             r'gis.*irrigat',r'irrigat.*gis',
             r'mapping.*yield',r'yield.*mapping',
             r'spatial.*yield',r'yield.*spatial',
             r'spatial.*crop',r'crop.*spatial']

CROP_MON  = [r'crop monitoring',r'crop classification',r'crop mapping',
             r'crop detect',r'yield predict',r'yield estimat',
             r'plant disease.*detect',r'detect.*plant disease',
             r'weed detect',r'weed map',r'weed manag',
             r'nitrogen.*crop',r'crop.*nitrogen',
             r'uav.*soil.*agri',r'biomass.*crop',r'crop.*biomass']

ALL_POS = FOSS + UAV_CROP + SAT_CROP + PREC_AGRI + GIS_CROP + CROP_MON

def has_positive(text):
    return any(re.search(t, text, re.IGNORECASE) for t in ALL_POS)

df_clean['_pos'] = df_clean['_text'].apply(has_positive)
df_qual = df_clean[df_clean['_pos']].copy()
print(f"Com sinal positivo: {len(df_qual)}")

# ── 4. Score de relevância temática ─────────────────────────────────────────
def rel_score(text):
    s = 0
    for t in FOSS:
        if re.search(t, text, re.IGNORECASE): s += 7
    for t in PREC_AGRI:
        if re.search(t, text, re.IGNORECASE): s += 5
    for t in UAV_CROP + SAT_CROP + GIS_CROP + CROP_MON:
        if re.search(t, text, re.IGNORECASE): s += 3
    return s

df_qual['_rel'] = df_qual['_text'].apply(rel_score)

# ── 5. Score composto ─────────────────────────────────────────────────────────
YEAR_REF = 2026
df_qual['_vel'] = df_qual['Citing Works Count'] / (YEAR_REF - df_qual['Publication Year']).clip(lower=1)

max_cit = df_qual['Citing Works Count'].max() or 1
max_vel = df_qual['_vel'].max() or 1
max_rel = df_qual['_rel'].max() or 1

df_qual['_score'] = (
    0.50 * (df_qual['Citing Works Count'] / max_cit) +
    0.30 * (df_qual['_vel'] / max_vel) +
    0.20 * (df_qual['_rel'] / max_rel)
)

df_top = df_qual.sort_values('_score', ascending=False).head(57).copy()
print(f"\n{'='*95}")
print(f"TOP 57 ARTIGOS FINAIS")
print(f"{'='*95}")

for i, (_, row) in enumerate(df_top.iterrows(), 1):
    foss = '[FOSS]' if any(re.search(t, row['_text'], re.IGNORECASE) for t in FOSS) else '[GIS ]'
    print(f"{i:2}. {foss} Cit={row['Citing Works Count']:4d} Vel={row['_vel']:5.1f} "
          f"Score={row['_score']:.3f} ({row['Publication Year']}) "
          f"{str(row['Title'])[:72]}")

print(f"\nPool qualificado total: {len(df_qual)}")

# ── 6. Exportar CSV com colunas em português ─────────────────────────────────
RENAME = {
    'Author/s':            'Autores',
    'Publication Year':    'Ano',
    'Title':               'Título',
    'Keywords':            'Palavras-chave',
    'Source Title':        'Revista',
    'Citing Works Count':  'Número de Citações',
    'DOI':                 'DOI',
    'Abstract':            'Abstract',
}

cols_orig  = list(RENAME.keys())
cols_exist = [c for c in cols_orig if c in df_top.columns]

df_out = df_top[cols_exist].rename(columns=RENAME).copy()
df_out.insert(0, 'Rank', range(1, len(df_out) + 1))

# Preencher valores ausentes
for col in df_out.columns:
    df_out[col] = df_out[col].fillna('N/D').astype(str).str.strip()
    df_out[col] = df_out[col].replace({'nan': 'N/D', '': 'N/D'})

OUT_PATH = '/home/sandbox/top57_foss4g_agricultura.csv'
df_out.to_csv(OUT_PATH, index=False, encoding='utf-8-sig', sep=';')
print(f"\nCSV salvo em: {OUT_PATH}")
print(f"Colunas: {list(df_out.columns)}")
print(f"Linhas : {len(df_out)}")