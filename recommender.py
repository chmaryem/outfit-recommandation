from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
import time

from settings import BUNDLE_PATH, DEVICE, IMG_DIR, TOPK_POOL, IMG_SIZE, CSV_PATH


torch.serialization.add_safe_globals([StandardScaler])


bundle = torch.load(BUNDLE_PATH, map_location=DEVICE, weights_only=False)

models_state = bundle["models_state"]
model_kwargs  = bundle["model_kwargs"]
cat_maps      = bundle["cat_maps"]
scaler: StandardScaler = bundle["scaler"]
CATEGORIES = bundle["categories"]         
IMG_COL    = bundle["img_col"]             

IMG_EXTS   = tuple(bundle["img_exts"])  
CSV = CSV_PATH


WEATHER_ALIASES = {
    "ensoleillé": "Clear",
    "ensoleille": "Clear",
    "soleil": "Clear",
    "clair": "Clear",
    "pluie": "Rain",
    "pluvieux": "Rain",
    "rain": "Rain",
    "clear": "Clear",
}

def normalize_weather(v: str) -> str:
    if v is None:
        return "Clear"
    s = str(v).strip().lower()
    return WEATHER_ALIASES.get(s, v)  # fallback to original if not in aliases

def safe_index(col: str, val: str) -> int:
    """
    Get categorical index with a safe fallback to the first available index.
    Avoids KeyError when the incoming value wasn't seen during training.
    """
    m = cat_maps[col]
    if val in m:
        return m[val]
    # try lowercase match if categories are cased
    lower_map = {k.lower(): v for k, v in m.items()}
    if str(val).lower() in lower_map:
        return lower_map[str(val).lower()]
    # last resort: take the first index
    return next(iter(m.values()))

# ----- data -----
def resolve_img_path(img_id: str|int) -> Path|None:
    x = str(img_id)
    if x.lower().endswith(IMG_EXTS):
        p = IMG_DIR / x
        return p if p.exists() else None
    for ext in IMG_EXTS:
        p = IMG_DIR / (x + ext)
        if p.exists(): return p
    return None

df = pd.read_csv(CSV)
df = df[df["semantic_category"].isin(CATEGORIES)].copy()
df["resolved_path"] = df[IMG_COL].apply(resolve_img_path)
df = df.dropna(subset=["resolved_path"]).reset_index(drop=True)
for col in ["emotion","weather_condition","preferred_style","preferred_preference"]:
    if col not in df.columns: df[col] = "unknown"
    df[col] = df[col].astype("category")

# ----- transforms -----
img_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----- model (same as train) -----
class ContextTower(nn.Module):
    def __init__(self, sizes, d):
        super().__init__()
        num_emotion, num_weather, num_style, num_pref = sizes
        self.emb_em = nn.Embedding(num_emotion, 24)
        self.emb_we = nn.Embedding(num_weather, 24)
        self.emb_st = nn.Embedding(num_style, 24)
        self.emb_pr = nn.Embedding(num_pref, 24)
        self.mlp = nn.Sequential(
            nn.Linear(24*4 + 1, 256), nn.ReLU(),
            nn.Linear(256, d)
        )
    def forward(self, cat_ids, temp_z):
        import torch.nn.functional as F
        e = torch.cat([
            self.emb_em(cat_ids[:,0]),
            self.emb_we(cat_ids[:,1]),
            self.emb_st(cat_ids[:,2]),
            self.emb_pr(cat_ids[:,3]),
            temp_z
        ], dim=1)
        return F.normalize(self.mlp(e), dim=1)

class ImageTower(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in self.backbone.parameters(): p.requires_grad=False
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(2048, d)
    def forward(self, x):
        import torch.nn.functional as F
        return F.normalize(self.proj(self.backbone(x)), dim=1)

class MatchModel(nn.Module):
    def __init__(self, sizes, d):
        super().__init__()
        self.ctx = ContextTower(sizes, d=d)
        self.img = ImageTower(d=d)
    def score(self, ctx_ids, temp_z, imgs):
        zc = self.ctx(ctx_ids, temp_z)
        zi = self.img(imgs)
        return (zc * zi).sum(dim=1)

sizes = tuple(model_kwargs["sizes"])
EMB_D = int(model_kwargs["emb_d"])

# instantiate & load all category models
models_by_cat: dict[str, MatchModel] = {}
for cat, state in models_state.items():
    m = MatchModel(sizes, d=EMB_D).to(DEVICE)
    m.load_state_dict(state, strict=True)
    m.eval()
    models_by_cat[cat] = m

# ---------- smart rules covering all categories ----------
GOOD = {
    "Rain": {
        "tops": ["hoodie","windbreaker","raincoat","parka","jacket","shell","waterproof"],
        "bottoms": ["jogger","track","cargo","waterproof","jeans"],
        "shoes": ["sneaker","trainer","running","trail","boot","waterproof","gore-tex","chelsea"],
        "bags": ["backpack","crossbody","messenger","nylon","waterproof"],
        "hats": ["cap","bucket","beanie","waterproof","rain"],
        "scarves": ["wool","cashmere","knit"],
        "jewellery": [],
        "sunglasses": [],
        "all-body": ["raincoat","poncho","overall"]
    },
    "Cold": {
        "tops": ["hoodie","sweat","sweater","knit","parka","coat","jacket","fleece"],
        "bottoms": ["thermal","wool","corduroy","lined"],
        "shoes": ["boot","high-top","insulated","fur-lined"],
        "bags": ["backpack","crossbody"],
        "hats": ["beanie","wool","knit","fur","beret"],
        "scarves": ["wool","cashmere","knit","pashmina"],
        "jewellery": [],
        "sunglasses": ["polarized"],
        "all-body": ["coat","parka","puffer"]
    },
    "Clear": {
        "tops": ["tee","t-shirt","tank","linen","crop"],
        "bottoms": ["short","skirt","linen","culotte"],
        "shoes": ["sandal","espadrille","flip-flop","canvas"],
        "bags": ["tote","crossbody","canvas","beach"],
        "hats": ["cap","sunhat","visor","bucket"],
        "scarves": ["silk","lightweight"],
        "jewellery": ["bracelet","anklet","light","minimal"],
        "sunglasses": ["aviator","wayfarer","round","cat-eye"],
        "all-body": ["summer","linen","dress","romper","jumpsuit"]
    },
    "sport": ["hoodie","sweat","track","running","trainer","sneaker","jogger",
              "windbreaker","parka","athletic","leggings","sport"]
}
BAD = {
    "Rain": ["sandal","open","flip-flop","heel","stiletto","suede","clutch","evening","sequin","sunglass","straw","canvas"],
    "Cold": ["sandal","flip-flop","open","crop","linen","mesh"],
    "Clear": ["parka","coat","wool","fur","thermal","fleece","heavy"],
    "generic": ["evening","formal","wedding","stiletto","sequin","gown","ball","tuxedo"]
}

def passes_hard_filters(name: str, cat: str, weather: str) -> bool:
    n = name.lower()
    if weather.lower() == "rain":
        if cat == "shoes" and any(k in n for k in ["heel","stiletto","sandal","open","flip-flop"]): return False
        if cat == "bags"  and any(k in n for k in ["clutch","evening"]): return False
        if cat == "sunglasses": return False
    return True

def keyword_score(name: str, cat: str, weather: str, style: str, temp_c: float) -> float:
    n = name.lower()
    bonus = 0.0
    # weather bonuses
    for k in GOOD.get(weather, {}).get(cat, []):
        if k in n: bonus += 0.35
    # sport style
    if style.lower() == "sport":
        for k in GOOD["sport"]:
            if k in n: bonus += 0.35
    # temp bumps
    if temp_c < 12:
        for k in ["Cold"]:
            for kk in GOOD[k].get(cat, []):
                if kk in n: bonus += 0.25
    if temp_c > 26:
        for k in ["Clear"]:
            for kk in GOOD[k].get(cat, []):
                if kk in n: bonus += 0.2
    # penalties
    for k in BAD["generic"]:
        if k in n: bonus -= 0.3
    for k in BAD.get(weather, []):
        if k in n: bonus -= 0.6
    return bonus

def to_temp_z(temp_value) -> float:
    z = scaler.transform(pd.DataFrame({"temperature": [float(temp_value)]}))
    return float(z.ravel()[0])

# ----- public API for FastAPI -----
@torch.no_grad()
def recommend(context: dict) -> dict:
    """
    context: {
      "emotion": str,
      "weather_condition": str,
      "temperature": float,
      "preferred_style": str,
      "preferred_preference": str
    }
    returns: {category: {"item_name": str, "image_path": str}}
    """
    t0 = time.time()
    print(f"[reco] START context={context}")

    # build a pseudo-row for encoding
    row = df.iloc[0].copy()
    for k, v in context.items():
        row[k] = v

    # normalize cross-system fields
    row["weather_condition"] = normalize_weather(row.get("weather_condition", "Clear"))
    row["temperature_z"] = to_temp_z(row["temperature"])

    # encode
    ids = torch.tensor([[ 
        safe_index("emotion", row.get("emotion", "unknown")),
        safe_index("weather_condition", row.get("weather_condition", "Clear")),
        safe_index("preferred_style", row.get("preferred_style", "unknown")),
        safe_index("preferred_preference", row.get("preferred_preference", "unknown")),
    ]], dtype=torch.long).to(DEVICE)
    
    temp = torch.tensor([[row["temperature_z"]]], dtype=torch.float).to(DEVICE)

    weather = str(row["weather_condition"])
    style   = str(row["preferred_style"])
    temp_c  = float(row["temperature"])

    out = {}
    for cat, model in models_by_cat.items():
        t_cat = time.time()
        pool = df[df["semantic_category"] == cat]
        if len(pool) == 0: 
            continue

        # hard filters
        pool_f = pool[pool["items"].astype(str).apply(lambda n: passes_hard_filters(n, cat, weather))]
        if len(pool_f) == 0:
            pool_f = pool

        if len(pool_f) > TOPK_POOL:
            pool_f = pool_f.sample(TOPK_POOL, random_state=42)

        print(f"[reco] {cat}: scoring {len(pool_f)} items")

        best_s, best_row = -1e9, None
        for _, r in pool_f.iterrows():
            img = Image.open(r["resolved_path"]).convert("RGB")
            img_t = img_tf(img).unsqueeze(0).to(DEVICE)
            s = model.score(ids, temp, img_t).item()
            s += keyword_score(str(r["items"]), cat, weather, style, temp_c)
            if s > best_s:
                best_s, best_row = s, r

        if best_row is not None:
            out[cat] = {
                "item_name": str(best_row["items"]),
                "image_path": str(best_row["resolved_path"])
            }
        print(f"[reco] {cat} done in {time.time() - t_cat:.2f}s")

    print(f"[reco] TOTAL time = {time.time() - t0:.2f}s")
    return out
