import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import scipy.stats as stats
from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]
from PIL import Image

# ── Konfigurasi ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TomatoScan",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:     #0a0f0a;
    --bg2:    #111811;
    --bg3:    #1a2a1a;
    --card:   #1a231a;
    --border: #2a3d2a;
    --green:  #4ade80;
    --green2: #22c55e;
    --green3: #16a34a;
    --dim:    #86efac;
    --muted:  #7a9e7a;
    --text:   #eef5ee;
    --text2:  #c4d9c4;
    --red:    #f87171;
    --orange: #fb923c;
    --yellow: #fbbf24;
}
* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem !important; max-width: 1400px; }

.navbar {
    display:flex; align-items:center; justify-content:space-between;
    padding:20px 0 28px; border-bottom:1px solid var(--border); margin-bottom:48px;
}
.navbar-brand { display:flex; align-items:center; gap:12px; }
.navbar-logo {
    width:36px; height:36px;
    background:linear-gradient(135deg,var(--green3),var(--green));
    border-radius:10px; display:flex; align-items:center;
    justify-content:center; font-size:1.2rem;
}
.navbar-title { font-family:'Playfair Display',serif; font-size:1.35rem; color:var(--text); letter-spacing:-0.02em; }
.navbar-sub { font-size:0.72rem; color:var(--text2); letter-spacing:0.12em; text-transform:uppercase; font-weight:600; }
.navbar-pill {
    background:rgba(74,222,128,0.08); border:1px solid rgba(74,222,128,0.2);
    color:var(--green); font-size:0.72rem; font-weight:600;
    padding:6px 14px; border-radius:100px; letter-spacing:0.06em;
}

.hero {
    position:relative;
    background:linear-gradient(135deg,#0d1f0d 0%,#111f11 50%,#0a1a0a 100%);
    border:1px solid var(--border); border-radius:24px;
    padding:64px 56px; margin-bottom:32px; overflow:hidden;
}
.hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(74,222,128,0.08) 0%,transparent 70%);
}
.hero::after {
    content:'🍅'; position:absolute; font-size:180px;
    right:40px; bottom:-20px; opacity:0.06; line-height:1;
}
.hero-eyebrow {
    font-size:0.72rem; font-weight:700; letter-spacing:0.15em;
    text-transform:uppercase; color:var(--green); margin-bottom:16px;
    display:flex; align-items:center; gap:8px;
}
.hero-eyebrow::before { content:''; width:24px; height:2px; background:var(--green); border-radius:1px; }
.hero h1 {
    font-family:'Playfair Display',serif; font-size:3.4rem;
    line-height:1.08; letter-spacing:-0.03em; color:var(--text);
    margin:0 0 20px; max-width:600px;
}
.hero h1 em { color:var(--green); font-style:italic; }
.hero p { font-size:1rem; color:var(--text2); line-height:1.75; max-width:520px; margin:0; }

.stat-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:48px; }
.stat-item {
    background:var(--card); border:1px solid var(--border);
    border-radius:14px; padding:20px 22px; position:relative; overflow:hidden;
}
.stat-item::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,var(--green3),transparent);
}
.stat-num { font-family:'Playfair Display',serif; font-size:2rem; color:var(--text); line-height:1; margin-bottom:4px; }
.stat-label { font-size:0.75rem; color:var(--text2); font-weight:500; letter-spacing:0.04em; }

.sec { display:flex; align-items:center; gap:12px; margin-bottom:20px; }
.sec-line { width:3px; height:22px; background:linear-gradient(180deg,var(--green),transparent); border-radius:2px; }
.sec-title { font-family:'Playfair Display',serif; font-size:1.5rem; color:var(--text); letter-spacing:-0.02em; }
.sec-sub { font-size:0.85rem; color:var(--text2); margin:-12px 0 24px 15px; line-height:1.6; }

.how-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:48px; }
.how-card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:28px 24px; }
.how-num { font-family:'Playfair Display',serif; font-size:3.5rem; color:#3a5a3a; line-height:1; margin-bottom:16px; letter-spacing:-0.04em; }
.how-card h4 { font-size:0.95rem; font-weight:700; color:var(--text); margin:0 0 8px; }
.how-card p { font-size:0.83rem; color:var(--text2); margin:0; line-height:1.65; }
.how-tag {
    display:inline-block; background:rgba(74,222,128,0.08); color:var(--green);
    font-size:0.68rem; font-weight:700; letter-spacing:0.08em;
    padding:3px 10px; border-radius:100px; margin-bottom:12px;
    border:1px solid rgba(74,222,128,0.15);
}

.disease-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:48px; }
.dc { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px 16px; transition:all 0.2s; }
.dc:hover { border-color:var(--green3); background:#1f2e1f; transform:translateY(-2px); }
.dc-icon { width:32px; height:32px; margin-bottom:10px; display:block; }
.dc-name { font-size:0.82rem; font-weight:700; color:var(--text); margin-bottom:3px; }
.dc-type { font-size:0.7rem; color:var(--text2); margin-bottom:8px; }
.dc-desc { font-size:0.73rem; color:var(--text2); line-height:1.5; margin-bottom:10px; }
.sev { display:inline-block; font-size:0.65rem; font-weight:700; padding:2px 8px; border-radius:100px; letter-spacing:0.04em; }
.sev-st { background:rgba(248,113,113,0.15); color:#f87171; border:1px solid rgba(248,113,113,0.2); }
.sev-t  { background:rgba(251,146,60,0.15);  color:#fb923c; border:1px solid rgba(251,146,60,0.2); }
.sev-s  { background:rgba(251,191,36,0.15);  color:#fbbf24; border:1px solid rgba(251,191,36,0.2); }
.sev-n  { background:rgba(74,222,128,0.15);  color:#4ade80; border:1px solid rgba(74,222,128,0.2); }

.upload-zone {
    background:var(--card); border:2px dashed var(--border);
    border-radius:20px; padding:36px; text-align:center; margin-bottom:24px;
}
.upload-icon { font-size:2.5rem; margin-bottom:12px; }
.upload-title { font-size:1rem; font-weight:600; color:var(--text); margin-bottom:6px; }
.upload-sub { font-size:0.82rem; color:var(--text2); }

.res-card { background:var(--card); border:1px solid var(--border); border-radius:20px; padding:28px; margin-bottom:16px; }
.res-eyebrow { font-size:0.68rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--text2); margin-bottom:6px; }
.res-class { font-family:'Playfair Display',serif; font-size:2rem; color:var(--green); margin-bottom:6px; line-height:1.1; letter-spacing:-0.02em; }
.bar-bg { background:var(--bg3); border-radius:100px; height:8px; overflow:hidden; margin:8px 0 4px; }
.bar-fg { background:linear-gradient(90deg,var(--green3),var(--green)); height:100%; border-radius:100px; }
.conf-num { font-family:'Playfair Display',serif; font-size:1.6rem; color:var(--text); margin-bottom:18px; }

.top3-item { background:var(--bg3); border-radius:10px; padding:10px 14px; margin-bottom:8px; border:1px solid transparent; }
.top3-item.active { border-color:var(--green3); background:rgba(22,163,74,0.08); }

.info-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:12px; }
.info-card { background:var(--bg3); border-radius:14px; padding:18px 20px; border-left:3px solid var(--green3); }
.info-card.warn { border-left-color:var(--orange); }
.info-card.danger { border-left-color:var(--red); }
.info-card.neutral { border-left-color:var(--muted); }
.info-card h5 { font-size:0.72rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; color:var(--green); margin:0 0 6px; }
.info-card.warn h5 { color:var(--orange); }
.info-card.danger h5 { color:var(--red); }
.info-card.neutral h5 { color:var(--muted); }
.info-card p { font-size:0.83rem; color:var(--text2); margin:0; line-height:1.6; }

.entropy-bar {
    background:var(--bg3); border-radius:12px; padding:14px 18px; margin-top:8px;
    font-size:0.85rem; display:flex; align-items:center; gap:10px;
    border:1px solid var(--border);
}

.gc-steps { display:flex; flex-direction:column; gap:10px; margin-bottom:28px; }
.gc-step {
    background:var(--bg3); border-radius:12px; padding:14px 18px;
    border-left:3px solid var(--green3); display:flex; gap:16px; align-items:flex-start;
}
.gc-step-num { font-family:'Playfair Display',serif; font-size:1.4rem; color:#4a7a4a; min-width:24px; line-height:1.2; font-weight:700; }
.gc-step h5 { font-size:0.85rem; font-weight:700; color:var(--text); margin:0 0 3px; }
.gc-step p  { font-size:0.79rem; color:var(--text2); margin:0; line-height:1.55; }

.hm-legend { background:#0d150d; border-radius:14px; padding:20px; margin-top:14px; border:1px solid var(--border); }
.hm-legend-title { font-size:0.72rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:var(--dim); margin-bottom:14px; }
.hm-legend-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:12px; }
.hm-item { background:var(--bg3); border-radius:10px; padding:10px 14px; }
.hm-dot { font-size:1rem; font-weight:800; }
.hm-name { font-size:0.82rem; font-weight:600; color:var(--text); }
.hm-desc { font-size:0.73rem; color:var(--text2); margin-top:2px; }
.hm-tip { background:rgba(74,222,128,0.05); border-radius:10px; padding:10px 14px; font-size:0.8rem; color:var(--text2); border:1px solid rgba(74,222,128,0.1); line-height:1.55; }

.err-box { background:rgba(248,113,113,0.05); border:1px solid rgba(248,113,113,0.2); border-radius:16px; padding:22px 24px; color:#fca5a5; font-size:0.88rem; line-height:1.7; }
hr.div { border:none; border-top:1px solid var(--border); margin:44px 0; }
.footer { text-align:center; color:var(--text2); font-size:0.75rem; padding:24px 0 8px; letter-spacing:0.04em; }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

DISPLAY_NAMES = {
    "Tomato___Bacterial_spot":                       "Bacterial Spot",
    "Tomato___Early_blight":                         "Early Blight",
    "Tomato___Late_blight":                          "Late Blight",
    "Tomato___Leaf_Mold":                            "Leaf Mold",
    "Tomato___Septoria_leaf_spot":                   "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites",
    "Tomato___Target_Spot":                          "Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":        "Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus":                  "Mosaic Virus",
    "Tomato___healthy":                              "Healthy",
}

CLASS_INFO = {
    "Bacterial Spot":        {"icon":"<svg width='28' height='28' viewBox='0 0 28 28' fill='none'><circle cx='14' cy='14' r='13' fill='#3d1a0a' stroke='#8B4513' stroke-width='1.5'/><circle cx='10' cy='11' r='2.5' fill='#8B4513'/><circle cx='17' cy='9' r='2' fill='#6B3410'/><circle cx='14' cy='16' r='3' fill='#A0522D'/><circle cx='9' cy='17' r='1.5' fill='#8B4513'/><circle cx='19' cy='15' r='2' fill='#6B3410'/></svg>","sev":"Tinggi","sev_class":"sev-t","type":"Bakteri","gejala":"Bercak kecil coklat gelap dengan halo kuning. Dapat bergabung membentuk area yang lebih luas.","cause":"Bakteri Xanthomonas campestris. Menyebar lewat percikan air dan alat pertanian terkontaminasi.","treatment":"Bakterisida berbahan tembaga setiap 7–10 hari. Hindari penyiraman dari atas."},
    "Early Blight":          {"icon":"<svg width='28' height='28' viewBox='0 0 28 28' fill='none'><circle cx='14' cy='14' r='13' fill='#2a1500' stroke='#8B4513' stroke-width='1.5'/><circle cx='14' cy='14' r='9' fill='none' stroke='#A0522D' stroke-width='1.2'/><circle cx='14' cy='14' r='6' fill='none' stroke='#8B4513' stroke-width='1.2'/><circle cx='14' cy='14' r='3' fill='none' stroke='#6B3410' stroke-width='1.2'/><circle cx='14' cy='14' r='1' fill='#A0522D'/></svg>","sev":"Sedang","sev_class":"sev-s","type":"Jamur","gejala":"Bercak coklat tua berbentuk cincin-cincin konsentris. Dimulai dari daun tua bagian bawah.","cause":"Jamur Alternaria solani. Berkembang pada suhu 15–27°C dengan kelembaban tinggi.","treatment":"Fungisida mankozeb atau klorotalonil setiap 7–10 hari. Buang daun terinfeksi."},
    "Late Blight":           {"icon":"<svg width='28' height='28' viewBox='0 0 28 28' fill='none'><circle cx='14' cy='14' r='13' fill='#0d1a0d' stroke='#2d4a2d' stroke-width='1.5'/><path d='M7 10 Q10 7 14 9 Q18 7 21 10 Q22 14 20 18 Q17 22 14 21 Q11 22 8 18 Q6 14 7 10Z' fill='#1a3a1a' stroke='#2d5a2d' stroke-width='1'/><path d='M10 11 Q14 9 18 11 Q20 14 18 17 Q14 19 10 17 Q8 14 10 11Z' fill='#152a15'/><ellipse cx='14' cy='18' rx='4' ry='2' fill='rgba(200,220,200,0.15)'/></svg>","sev":"Sangat Tinggi","sev_class":"sev-st","type":"Oomycete","gejala":"Bercak basah hijau-abu gelap yang cepat berubah coklat kehitaman. Lapisan putih berbulu di bawah daun.","cause":"Phytophthora infestans. Menyebar cepat pada kondisi dingin 10–25°C dan lembab.","treatment":"Fungisida sistemik metalaksil. Cabut dan musnahkan tanaman terinfeksi segera."},
    "Leaf Mold":             {"icon":"<svg width='28' height='28' viewBox='0 0 28 28' fill='none'><circle cx='14' cy='14' r='13' fill='#1a2a0d' stroke='#4a7a2d' stroke-width='1.5'/><ellipse cx='14' cy='13' rx='7' ry='9' fill='#2a4a1a' stroke='#3a6a2a' stroke-width='1'/><circle cx='11' cy='11' r='2' fill='rgba(180,200,120,0.3)'/><circle cx='16' cy='13' r='1.5' fill='rgba(180,200,120,0.25)'/><circle cx='13' cy='16' r='2.5' fill='rgba(180,200,120,0.2)'/><path d='M14 6 Q14 4 14 3' stroke='#4a7a2a' stroke-width='1.5' stroke-linecap='round'/></svg>","sev":"Sedang","sev_class":"sev-s","type":"Jamur","gejala":"Bercak kuning pucat di atas daun, lapisan berbulu abu-abu di bawah daun.","cause":"Jamur Passalora fulva. Berkembang pada kelembaban >85% dan suhu 21–24°C.","treatment":"Tingkatkan sirkulasi udara. Fungisida klorotalonil atau mankozeb."},
    "Septoria Leaf Spot":    {"icon":"<svg width='28' height='28' viewBox='0 0 28 28' fill='none'><circle cx='14' cy='14' r='13' fill='#1e2a1e' stroke='#3a5a3a' stroke-width='1.5'/><circle cx='10' cy='12' r='2.8' fill='#c8d0c0' stroke='#5a4a2a' stroke-width='1'/><circle cx='10' cy='12' r='1' fill='#2a2a2a'/><circle cx='17' cy='10' r='2.2' fill='#c8d0c0' stroke='#5a4a2a' stroke-width='1'/><circle cx='17' cy='10' r='0.8' fill='#2a2a2a'/><circle cx='15' cy='17' r='2.5' fill='#c8d0c0' stroke='#5a4a2a' stroke-width='1'/><circle cx='15' cy='17' r='0.9' fill='#2a2a2a'/></svg>","sev":"Sedang","sev_class":"sev-s","type":"Jamur","gejala":"Bercak kecil 1–4mm, pusat putih/abu-abu, tepi coklat gelap, titik hitam di tengah.","cause":"Jamur Septoria lycopersici. Menyebar melalui percikan air.","treatment":"Fungisida mankozeb. Hindari penyiraman dari atas."},
    "Spider Mites":          {"icon":"<svg width='28' height='28' viewBox='0 0 28 28' fill='none'><circle cx='14' cy='14' r='13' fill='#2a1a0a' stroke='#8B3a10' stroke-width='1.5'/><ellipse cx='14' cy='13' rx='4' ry='5' fill='#8B2500'/><ellipse cx='14' cy='15' rx='3' ry='3.5' fill='#a03010'/><line x1='7' y1='11' x2='11' y2='13' stroke='#8B3a10' stroke-width='1.2' stroke-linecap='round'/><line x1='7' y1='15' x2='11' y2='14' stroke='#8B3a10' stroke-width='1.2' stroke-linecap='round'/><line x1='21' y1='11' x2='17' y2='13' stroke='#8B3a10' stroke-width='1.2' stroke-linecap='round'/><line x1='21' y1='15' x2='17' y2='14' stroke='#8B3a10' stroke-width='1.2' stroke-linecap='round'/><circle cx='12' cy='11' r='1' fill='#ff6060'/><circle cx='16' cy='11' r='1' fill='#ff6060'/></svg>","sev":"Sedang","sev_class":"sev-s","type":"Hama","gejala":"Daun berbintik putih/kuning halus (stippling). Pada infestasi berat muncul benang tipis di bawah daun.","cause":"Tungau Tetranychus urticae. Berkembang pesat pada kondisi panas dan kering.","treatment":"Mitisida abamektin atau spiromesifen. Predator alami Phytoseiulus persimilis."},
    "Target Spot":           {"icon":"<svg width='28' height='28' viewBox='0 0 28 28' fill='none'><circle cx='14' cy='14' r='13' fill='#1e1a0a' stroke='#6a5a2a' stroke-width='1.5'/><circle cx='14' cy='14' r='9' fill='none' stroke='#8B6914' stroke-width='1.5'/><circle cx='14' cy='14' r='6' fill='none' stroke='#A07820' stroke-width='1.5'/><circle cx='14' cy='14' r='3' fill='none' stroke='#8B6914' stroke-width='1.5'/><circle cx='14' cy='14' r='1.2' fill='#c8a840'/></svg>","sev":"Sedang","sev_class":"sev-s","type":"Jamur","gejala":"Bercak coklat bulat dengan pola cincin konsentris menyerupai target tembak, 1–2 cm.","cause":"Jamur Corynespora cassiicola. Berkembang pada suhu 25–30°C dengan kelembaban tinggi.","treatment":"Fungisida azoksistrobin atau difenokonazol. Jaga jarak tanam."},
    "Yellow Leaf Curl Virus":{"icon":"<svg width='28' height='28' viewBox='0 0 28 28' fill='none'><circle cx='14' cy='14' r='13' fill='#2a2500' stroke='#8B8000' stroke-width='1.5'/><path d='M8 18 Q9 12 12 10 Q14 8 16 10 Q18 12 17 16 Q16 19 14 20 Q12 21 10 19 Q8 18 8 18Z' fill='#5a5500' stroke='#8B8014' stroke-width='1'/><path d='M10 16 Q11 13 13 12' stroke='#a0a030' stroke-width='1' stroke-linecap='round'/><path d='M13 16 Q14 14 16 13' stroke='#a0a030' stroke-width='1' stroke-linecap='round'/><path d='M8 14 Q10 10 14 9' stroke='#d4d060' stroke-width='1.2' stroke-linecap='round' fill='none'/></svg>","sev":"Sangat Tinggi","sev_class":"sev-st","type":"Virus","gejala":"Daun menggulung ke atas, menguning dari pinggir, mengecil dan menebal. Tanaman kerdil.","cause":"Tomato Yellow Leaf Curl Virus (TYLCV) — ditularkan kutu kebul Bemisia tabaci.","treatment":"Kendalikan populasi Bemisia tabaci. Gunakan mulsa perak. Cabut tanaman terinfeksi."},
    "Mosaic Virus":          {"icon":"<svg width='28' height='28' viewBox='0 0 28 28' fill='none'><circle cx='14' cy='14' r='13' fill='#0d1a0d' stroke='#2a5a2a' stroke-width='1.5'/><rect x='7' y='7' width='6' height='6' rx='1' fill='#1a4a1a'/><rect x='15' y='7' width='6' height='6' rx='1' fill='#2a6a2a'/><rect x='7' y='15' width='6' height='6' rx='1' fill='#2a6a2a'/><rect x='15' y='15' width='6' height='6' rx='1' fill='#1a4a1a'/><rect x='11' y='11' width='6' height='6' rx='1' fill='#3a8a3a'/></svg>","sev":"Tinggi","sev_class":"sev-t","type":"Virus","gejala":"Pola mozaik hijau tua–hijau muda tidak beraturan. Daun keriting dan tepinya bergelombang.","cause":"Tomato Mosaic Virus (ToMV). Ditularkan secara mekanis melalui alat dan kontak antartanaman.","treatment":"Gunakan benih bersertifikat. Desinfeksi alat dengan natrium hipoklorit 10%."},
    "Healthy":               {"icon":"<svg width='28' height='28' viewBox='0 0 28 28' fill='none'><circle cx='14' cy='14' r='13' fill='#0d2a0d' stroke='#22c55e' stroke-width='1.5'/><path d='M14 6 Q14 4 14 3' stroke='#4ade80' stroke-width='2' stroke-linecap='round'/><ellipse cx='14' cy='15' rx='7' ry='9' fill='#1a4a1a' stroke='#22c55e' stroke-width='1'/><path d='M10 12 Q12 10 14 12 Q16 10 18 12' stroke='#4ade80' stroke-width='1' fill='none' stroke-linecap='round'/><path d='M10 15 Q12 13 14 15 Q16 13 18 15' stroke='#4ade80' stroke-width='1' fill='none' stroke-linecap='round'/><path d='M11 18 Q13 16 14 18 Q15 16 17 18' stroke='#4ade80' stroke-width='1' fill='none' stroke-linecap='round'/></svg>","sev":"Tidak Ada","sev_class":"sev-n","type":"Sehat","gejala":"Daun hijau segar dan merata. Tidak ada bercak, bintik, atau kelainan bentuk apapun.","cause":"—","treatment":"Pertahankan perawatan rutin: penyiraman teratur, pemupukan berimbang, monitoring mingguan."},
}

# ── Fungsi ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_tomato_model():
    model = load_model("tomato_model.keras")
    model(tf.zeros((1, 192, 192, 3)))
    return model

@st.cache_resource
def build_grad_model(_model):
    rescaling_layer = _model.layers[1]
    base_model      = _model.layers[2]
    gap_layer       = _model.layers[3]
    dense_layer     = _model.layers[4]
    output_layer    = _model.layers[6]
    inp = tf.keras.Input(shape=(192, 192, 3))
    x   = rescaling_layer(inp)
    base_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer("out_relu").output, base_model.output]
    )
    conv_out, mobilenet_features = base_extractor(x)
    gap_out   = gap_layer(mobilenet_features)
    dense_out = dense_layer(gap_out)
    preds     = output_layer(dense_out)
    return tf.keras.Model(inputs=inp, outputs=[conv_out, preds])

def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((192, 192))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0), np.array(img, dtype=np.uint8)
# =============================================================================
# Prediction Confidence Threshold
# =============================================================================
CONFIDENCE_THRESHOLD = 0.80

def evaluate_prediction_confidence(model, img_array):
    pred = model.predict(img_array, verbose=0)[0]

    confidence = float(np.max(pred))
    entropy = float(stats.entropy(pred))

    is_confident = confidence >= CONFIDENCE_THRESHOLD

    return is_confident, confidence, entropy, pred

def make_gradcam_heatmap(img_array, grad_model):
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)
        pred_index    = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads        = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    max_val      = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()

def overlay_heatmap(img_display, heatmap, alpha=0.4):
    h       = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (colored * alpha + img_display * (1 - alpha)).astype(np.uint8)
    return h, colored, overlay

# ═══════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════

# NAVBAR
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <div class="navbar-logo">🍅</div>
        <div>
            <div class="navbar-title">TomatoScan</div>
            <div class="navbar-sub">Deep Learning · XAI</div>
        </div>
    </div>
    <div class="navbar-pill">MobileNetV2 + GradCAM</div>
</div>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Sistem Klasifikasi Penyakit Tanaman</div>
    <h1>Kenali penyakit daun tomat dengan <em>kecerdasan buatan</em></h1>
    <p>Upload foto daun tomat — sistem akan mengklasifikasikan penyakit secara otomatis
    dan menjelaskan <strong style="color:#86efac">area mana</strong> yang menjadi
    dasar keputusan model menggunakan GradCAM.</p>
</div>
""", unsafe_allow_html=True)

# STATS
st.markdown("""
<div class="stat-strip">
    <div class="stat-item"><div class="stat-num">10</div><div class="stat-label">Kelas Penyakit</div></div>
    <div class="stat-item"><div class="stat-num">87.9%</div><div class="stat-label">Test Accuracy</div></div>
    <div class="stat-item"><div class="stat-num">192px</div><div class="stat-label">Ukuran Input</div></div>
    <div class="stat-item"><div class="stat-num">PlantVillage</div><div class="stat-label">Sumber Dataset</div></div>
</div>
""", unsafe_allow_html=True)

# HOW IT WORKS
st.markdown("""
<div class="sec"><div class="sec-line"></div><div class="sec-title">Cara Kerja Sistem</div></div>
<p class="sec-sub">Tiga tahap proses dari gambar masuk hingga hasil prediksi dan penjelasannya.</p>
<div class="how-grid">
    <div class="how-card">
        <div class="how-num">01</div>
        <div class="how-tag">INPUT</div>
        <h4>Upload & Pra-pemrosesan</h4>
        <p>Gambar daun tomat di-resize ke 192×192 piksel dan dinormalisasi agar sesuai format input yang digunakan saat model dilatih.</p>
    </div>
    <div class="how-card">
        <div class="how-num">02</div>
        <div class="how-tag">KLASIFIKASI</div>
        <h4>MobileNetV2 Transfer Learning</h4>
        <p>Model mengekstrak fitur visual — tekstur, warna, pola bercak — lalu menghasilkan skor probabilitas untuk 10 kelas penyakit via Dense softmax.</p>
    </div>
    <div class="how-card">
        <div class="how-num">03</div>
        <div class="how-tag">PENJELASAN</div>
        <h4>GradCAM Explainability</h4>
        <p>GradCAM menghitung gradien dari skor kelas terhadap peta aktivasi layer out_relu untuk menghasilkan heatmap penjelas keputusan model.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='div'>", unsafe_allow_html=True)

# DISEASE GALLERY
st.markdown("""
<div class="sec"><div class="sec-line"></div><div class="sec-title">Kelas Penyakit yang Dapat Diklasifikasi</div></div>
<p class="sec-sub">10 kondisi daun tomat yang dikenali sistem ini.</p>
""", unsafe_allow_html=True)

cards = "<div class='disease-grid'>"
for name, info in CLASS_INFO.items():
    cards += f"""<div class='dc'>
        <div class='dc-icon'>{info['icon']}</div>
        <div class='dc-name'>{name}</div>
        <div class='dc-type'>{info['type']}</div>
        <div class='dc-desc'>{info['gejala']}</div>
    </div>"""
cards += "</div>"
st.markdown(cards, unsafe_allow_html=True)

st.markdown("<hr class='div'>", unsafe_allow_html=True)

# PREDIKSI
st.markdown("""
<div class="sec"><div class="sec-line"></div><div class="sec-title">Klasifikasi Penyakit</div></div>
<p class="sec-sub">Upload foto daun tomat untuk memulai analisis.</p>
<div class="upload-zone">
    <div class="upload-icon">📷</div>
    <div class="upload-title">Upload Gambar Daun Tomat</div>
    <div class="upload-sub">JPG / PNG / JPEG &nbsp;·&nbsp; Pastikan daun terlihat jelas dengan pencahayaan cukup</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded is not None:
    with st.spinner("Memuat model..."):
        try:
            model      = load_tomato_model()
            grad_model = build_grad_model(model)
        except Exception as e:
            st.error(f"G~agal memuat model: {e}\nPastikan file `tomato_model.keras` ada di folder yang sama.")
            st.stop()

    pil_img = Image.open(uploaded)
    img_array, img_display = preprocess(pil_img)

    with st.spinner("Menganalisis gambar..."):
        is_confident, confidence, entropy, pred = evaluate_prediction_confidence(model,img_array)

    if not is_confident:
        col_e1, col_e2 = st.columns([1, 2])
        with col_e1:
            st.image(pil_img, use_container_width=True)
        with col_e2:
            st.warning(f"""### ⚠️ Hasil klasifikasi tidak dapat ditampilkan
            Model hanya memiliki tingkat keyakinan sebesar **{confidence*100:.2f}%**.
            Silakan pastikan bahwa:
            - Gambar merupakan daun tomat.
            - Daun terlihat jelas dan tidak terpotong.
            - Pencahayaan cukup.
            - Gambar tidak blur atau buram.
            Silakan unggah gambar lain dengan kualitas yang lebih baik.""")
        st.stop()

    pred_idx   = int(np.argmax(pred))
    pred_raw   = CLASS_NAMES[pred_idx]
    pred_class = DISPLAY_NAMES[pred_raw]
    info       = CLASS_INFO[pred_class]
    top3_idx   = pred.argsort()[-3:][::-1]
    top3       = [(DISPLAY_NAMES[CLASS_NAMES[i]], float(pred[i])) for i in top3_idx]

    with st.spinner("Membuat heatmap GradCAM..."):
        heatmap                           = make_gradcam_heatmap(img_array, grad_model)
        heatmap_resized, heatmap_colored, overlay = overlay_heatmap(img_display, heatmap)

    col_img, col_res = st.columns([1, 1.5], gap="large")

    with col_img:
        st.image(pil_img, caption="Gambar yang diupload", use_container_width=True)

    with col_res:
        conf_pct = int(confidence * 100)
        if confidence < 0.85:
            st.info("ℹ️ Model berhasil melakukan klasifikasi, namun tingkat keyakinannya masih sedang. "
            "Gunakan gambar dengan pencahayaan yang baik agar hasil lebih akurat.")
        st.markdown(f"""
        <div class='res-card'>
            <div class='res-eyebrow'>Hasil Klasifikasi</div>
            <div class='res-class'>{pred_class}</div>
            <div style='font-size:0.83rem;color:var(--text2);margin-bottom:16px;line-height:1.6;'>
                {info['gejala'][:130]}...
            </div>
            <div style='font-size:0.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--text2);margin-bottom:4px;'>
                Confidence Score
            </div>
            <div class='bar-bg'><div class='bar-fg' style='width:{conf_pct}%'></div></div>
            <div class='conf-num'>{conf_pct}%</div>
            <div style='font-size:0.7rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--text2);margin-bottom:10px;'>
                Top 3 Prediksi
            </div>
        </div>""", unsafe_allow_html=True)

        for i, (cls, sc) in enumerate(top3):
            bw  = int(sc * 100)
            top = i == 0
            st.markdown(f"""
            <div class='top3-item {"active" if top else ""}'>
                <div style='display:flex;justify-content:space-between;align-items:center;
                    font-size:0.85rem;font-weight:{"600" if top else "400"};
                    color:{"var(--green)" if top else "var(--text2)"};margin-bottom:5px;'>
                    <span>{"🥇" if i==0 else "🥈" if i==1 else "🥉"} {cls}</span>
                    <span style='font-family:Playfair Display,serif;'>{bw}%</span>
                </div>
                <div style='background:var(--border);border-radius:100px;height:4px;overflow:hidden;'>
                    <div style='width:{bw}%;background:{"var(--green)" if top else "var(--muted)"};height:100%;border-radius:100px;'></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if pred_class != "Healthy":
        st.markdown(f"""
        <div class='info-grid'>
            <div class='info-card'><h5>🔬 Gejala</h5><p>{info['gejala']}</p></div>
            <div class='info-card warn'><h5>🦠 Penyebab</h5><p>{info['cause']}</p></div>
            <div class='info-card danger'><h5>💊 Penanganan</h5><p>{info['treatment']}</p></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='info-card' style='margin-bottom:14px;'>
            <h5>✅ Daun Sehat</h5>
            <p>{info['gejala']}<br><br><strong>Saran:</strong> {info['treatment']}</p>
        </div>""", unsafe_allow_html=True)

    if entropy < 1:
        ent_icon, ent_msg, ent_color = "✅", "Model sangat yakin dengan prediksi ini.", "var(--green)"
    elif entropy < 2:
        ent_icon, ent_msg, ent_color = "⚠️", "Model cukup yakin — ada kemungkinan kecil kelas lain.", "var(--yellow)"
    else:
        ent_icon, ent_msg, ent_color = "❌", "Model kurang yakin — gambar mungkin ambigu.", "var(--red)"

    st.markdown(f"""
    <div class='entropy-bar'>
        <span style='font-size:1.1rem;'>{ent_icon}</span>
        <span style='color:{ent_color};font-weight:600;font-size:0.86rem;'>{ent_msg}</span>
        <span style='color:var(--text2);font-size:0.82rem;margin-left:auto;'>Entropy: {entropy:.3f}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # GRADCAM
    st.markdown("""
    <div class="sec"><div class="sec-line"></div><div class="sec-title">Explainable AI — GradCAM</div></div>
    <p class="sec-sub">Visualisasi area gambar yang paling memengaruhi keputusan model.</p>
    <div class='gc-steps'>
        <div class='gc-step'><div class='gc-step-num'>①</div>
            <div><h5>Forward Pass</h5><p>Gambar diumpankan ke seluruh layer CNN hingga menghasilkan skor prediksi untuk tiap kelas.</p></div></div>
        <div class='gc-step'><div class='gc-step-num'>②</div>
            <div><h5>Hitung Gradien</h5><p>Dihitung gradien dari skor kelas terhadap peta aktivasi layer out_relu — nilai besar berarti area tersebut sangat berpengaruh ke prediksi.</p></div></div>
        <div class='gc-step'><div class='gc-step-num'>③</div>
            <div><h5>Global Average Pooling pada Gradien</h5><p>Gradien dirata-ratakan per channel untuk mendapat bobot (α) yang menyatakan kepentingan tiap channel terhadap kelas yang diprediksi.</p></div></div>
        <div class='gc-step'><div class='gc-step-num'>④</div>
            <div><h5>Weighted Sum + ReLU</h5><p>Peta aktivasi dikalikan bobotnya lalu dijumlahkan. ReLU hanya menyimpan nilai positif — area yang mendukung prediksi.</p></div></div>
        <div class='gc-step'><div class='gc-step-num'>⑤</div>
            <div><h5>Resize & Overlay</h5><p>Heatmap kecil (6×6) di-resize ke 192×192, diberi warna JET, lalu ditempel ke gambar original dengan transparansi 40%.</p></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.78rem;font-weight:600;color:var(--text2);letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px;'>Hasil Visualisasi</p>", unsafe_allow_html=True)

    gc1, gc2, gc3 = st.columns(3, gap="medium")
    with gc1:
        st.image(img_display, use_container_width=True)
        st.markdown("<p style='text-align:center;font-size:0.75rem;color:var(--text2);margin-top:6px;'>① Gambar Original</p>", unsafe_allow_html=True)
    with gc2:
        fig_hm, ax_hm = plt.subplots(figsize=(5, 5))
        fig_hm.patch.set_facecolor("#0a0f0a")
        ax_hm.set_facecolor("#0a0f0a")
        ax_hm.imshow(heatmap_resized, cmap="jet")
        ax_hm.axis("off")
        st.pyplot(fig_hm)
        plt.close(fig_hm)
        st.markdown("<p style='text-align:center;font-size:0.75rem;color:var(--text2);margin-top:6px;'>② Heatmap GradCAM</p>", unsafe_allow_html=True)
    with gc3:
        st.image(overlay, use_container_width=True)
        st.markdown("<p style='text-align:center;font-size:0.75rem;color:var(--text2);margin-top:6px;'>③ Overlay</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='hm-legend'>
        <div class='hm-legend-title'>Panduan Membaca Heatmap</div>
        <div class='hm-legend-grid'>
            <div class='hm-item'>
                <div class='hm-dot' style='color:#f87171;'>■</div>
                <div class='hm-name'>Merah / Oranye</div>
                <div class='hm-desc'>Area paling penting — fokus utama model</div>
            </div>
            <div class='hm-item'>
                <div class='hm-dot' style='color:#fbbf24;'>■</div>
                <div class='hm-name'>Kuning / Hijau</div>
                <div class='hm-desc'>Cukup penting, diperhatikan sebagian</div>
            </div>
            <div class='hm-item'>
                <div class='hm-dot' style='color:#60a5fa;'>■</div>
                <div class='hm-name'>Biru</div>
                <div class='hm-desc'>Diabaikan model, tidak relevan</div>
            </div>
        </div>
        <div class='hm-tip'>
            💡 <strong>Idealnya</strong> area merah berada tepat di atas bercak atau lesi pada daun.
            Jika area merah justru di background, model mungkin terpengaruh pola di luar daun saat training.
        </div>
    </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:64px;color:var(--text2);'>
        <div style='font-size:4rem;margin-bottom:16px;opacity:0.4;'>🍃</div>
        <div style='font-size:1rem;font-weight:500;color:var(--text2);'>Upload foto daun tomat untuk memulai klasifikasi</div>
        <div style='font-size:0.82rem;margin-top:8px;'>Sistem akan menganalisis dan menjelaskan hasil prediksinya</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<hr class='div'>
<div class='footer'>
    TomatoScan &nbsp;·&nbsp; Tugas Akhir &nbsp;·&nbsp;
    Klasifikasi Penyakit Daun Tomat dengan Deep Learning & XAI<br>
    Dataset: PlantVillage &nbsp;·&nbsp; Arsitektur: MobileNetV2 &nbsp;·&nbsp; XAI: GradCAM
</div>""", unsafe_allow_html=True)
