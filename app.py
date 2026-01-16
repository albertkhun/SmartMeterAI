from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, Query
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import pandas as pd
import os
import io

from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from database.db import engine, Base
from database.deps import get_db
from models.models import Upload, UsageRecord, ConsumerRisk, Feedback

from services.preprocess import preprocess_data
from services.features import make_features
from services.riskmodel import run_hybrid_risk_engine
from services.explain import build_reason_text
from services.layers import compute_layer_scores, compute_final_risk
from services.payment_feature import add_payment_features, compute_cross_behavior_risk


# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Smart Meter Theft Risk System (Hybrid Gov-Grade)")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv", ".txt"}

# Store last results for report download
latest_results_df = None


def allowed_file(filename: str):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


def read_any_file(filepath: str):
    filepath_lower = filepath.lower()

    if filepath_lower.endswith(".csv"):
        return pd.read_csv(filepath)

    if filepath_lower.endswith(".tsv") or filepath_lower.endswith(".txt"):
        return pd.read_csv(filepath, sep="\t")

    if filepath_lower.endswith(".xlsx"):
        return pd.read_excel(filepath, engine="openpyxl")

    if filepath_lower.endswith(".xls"):
        return pd.read_excel(filepath, engine="xlrd")

    raise ValueError("Unsupported file format")


def risk_band(score: float) -> str:
    try:
        score = float(score)
    except:
        score = 0.0

    if score >= 85:
        return "Critical"
    elif score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"


# -------------------------
# Upload Page
# -------------------------
@app.get("/")
def upload_page(request: Request):
    
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    global latest_results_df

    if not allowed_file(file.filename):
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "msg": "❌ Unsupported file type! Upload CSV/XLSX/XLS/TSV/TXT only."
        })

    filepath = os.path.join(UPLOAD_DIR, file.filename)

    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Save upload info
    db.add(Upload(filename=file.filename))
    db.commit()

    # Read + preprocess
    df = read_any_file(filepath)
    df = preprocess_data(df)

    # Store usage records into DB (reset each upload)
    db.query(UsageRecord).delete()
    db.commit()

    for _, row in df.iterrows():
        db.add(UsageRecord(
            consumer_id=str(row["consumer_id"]),
            timestamp=row["timestamp"],
            kwh=float(row["kwh"])
        ))
    db.commit()

    # Feature engineering
    features = make_features(df)

    # Run Hybrid AI Engine (Rules + IF + LOF + Ensemble)
    results = run_hybrid_risk_engine(features)

    # Add explainable reason
    results["reason"] = results.apply(build_reason_text, axis=1)

    # -------------------------
    # Add 6-Layer scoring
    # -------------------------
    layer_rows = []
    final_scores = []

    for _, row in results.iterrows():
        layers = compute_layer_scores(row)
        layer_rows.append(layers)
        final_scores.append(compute_final_risk(layers))

    layers_df = pd.DataFrame(layer_rows)
    results = pd.concat([results.reset_index(drop=True), layers_df.reset_index(drop=True)], axis=1)

    # Override risk_score with layered score
    results["risk_score"] = final_scores

    # Add risk band
    results["risk_band"] = results["risk_score"].apply(risk_band)

    # Save latest results for report download
    latest_results_df = results.copy()

    # Store consumer risk into DB (reset each upload)
    db.query(ConsumerRisk).delete()
    db.commit()

    for _, r in results.iterrows():
        db.add(ConsumerRisk(
            consumer_id=str(r["consumer_id"]),
            risk_score=float(r["risk_score"]),
            label=str(r["label"]),
            reason=str(r["reason"]),
            action=str(r["action"])
        ))
    db.commit()

    preview = df.head(5).to_dict(orient="records")

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "msg": "✅ File uploaded, processed, and Hybrid + 6-Layer AI results stored successfully!",
        "preview": preview
    })


# -------------------------
# Dashboard
# -------------------------
@app.get("/dashboard")
def dashboard(request: Request, db: Session = Depends(get_db)):
    total_consumers = db.query(ConsumerRisk).count()
    total_records = db.query(UsageRecord).count()

    normal = db.query(ConsumerRisk).filter(ConsumerRisk.label == "Normal").count()
    monitor = db.query(ConsumerRisk).filter(ConsumerRisk.label == "Monitor").count()
    inspection = db.query(ConsumerRisk).filter(ConsumerRisk.label == "Inspection").count()

    confirmed = db.query(ConsumerRisk).filter(ConsumerRisk.label == "Confirmed Theft").count()
    cleared = db.query(ConsumerRisk).filter(ConsumerRisk.label == "Cleared").count()
    pending = db.query(ConsumerRisk).filter(ConsumerRisk.label == "Pending Verification").count()

    top_risky = (
        db.query(ConsumerRisk)
        .order_by(ConsumerRisk.risk_score.desc())
        .limit(8)
        .all()
    )

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "total_consumers": total_consumers,
        "total_records": total_records,
        "normal": normal,
        "monitor": monitor,
        "inspection": inspection,
        "confirmed": confirmed,
        "cleared": cleared,
        "pending": pending,
        "top_risky": top_risky
    })


# -------------------------
# Alerts Page
# -------------------------
@app.get("/alerts")
def alerts(
    request: Request,
    filter: str = Query("all"),
    db: Session = Depends(get_db)
):
    query = db.query(ConsumerRisk)

    if filter == "inspection":
        query = query.filter(ConsumerRisk.label == "Inspection")

    elif filter == "monitor":
        query = query.filter(ConsumerRisk.label == "Monitor")

    elif filter == "normal":
        query = query.filter(ConsumerRisk.label == "Normal")

    elif filter == "critical":
        query = query.filter(ConsumerRisk.risk_score >= 85)

    elif filter == "high":
        query = query.filter(ConsumerRisk.risk_score.between(70, 84.99))

    elif filter == "medium":
        query = query.filter(ConsumerRisk.risk_score.between(40, 69.99))

    elif filter == "low":
        query = query.filter(ConsumerRisk.risk_score < 40)

    elif filter == "confirmed":
        query = query.filter(ConsumerRisk.label == "Confirmed Theft")

    rows = query.order_by(ConsumerRisk.risk_score.desc()).all()

    return templates.TemplateResponse("alerts.html", {
        "request": request,
        "rows": rows,
        "filter": filter
    })


# -------------------------
# Consumer Detail
# -------------------------
@app.get("/consumer/{consumer_id}")
def consumer_detail(consumer_id: str, request: Request, db: Session = Depends(get_db)):
    risk = db.query(ConsumerRisk).filter(ConsumerRisk.consumer_id == consumer_id).first()

    records = (
        db.query(UsageRecord)
        .filter(UsageRecord.consumer_id == consumer_id)
        .order_by(UsageRecord.timestamp.asc())
        .all()
    )

    timestamps = [r.timestamp.strftime("%Y-%m-%d") for r in records]
    kwh_values = [r.kwh for r in records]

    return templates.TemplateResponse("consumers.html", {
        "request": request,
        "risk": risk,
        "consumer_id": consumer_id,
        "timestamps": timestamps,
        "kwh_values": kwh_values
    })


# -------------------------
# Review Page (Human-in-loop)
# -------------------------
@app.get("/review")
def review_page(request: Request):
    return templates.TemplateResponse("review.html", {"request": request})


@app.get("/review")
def review_page(request: Request, db: Session = Depends(get_db)):
    consumers = (
        db.query(ConsumerRisk.consumer_id)
        .order_by(ConsumerRisk.risk_score.desc())
        .all()
    )

    consumer_ids = [c[0] for c in consumers]

    return templates.TemplateResponse("review.html", {
        "request": request,
        "consumer_ids": consumer_ids
    })


# -------------------------
# Download Inspection Report (CSV)
# -------------------------
@app.get("/download-report")
def download_report():
    global latest_results_df

    if latest_results_df is None or len(latest_results_df) == 0:
        return {"error": "No report available. Upload a file first."}

    report_df = latest_results_df.copy()

    cols = [
        "consumer_id",
        "risk_score",
        "risk_band",
        "label",
        "action",
        "confidence",
        "reason",
        "layer1_physics",
        "layer2_rules",
        "layer3_collusion",
        "layer4_ml",
        "layer5_evidence",
        "layer6_human",
    ]
    cols = [c for c in cols if c in report_df.columns]
    report_df = report_df[cols].sort_values("risk_score", ascending=False)

    stream = io.StringIO()
    report_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=inspection_report.csv"},
    )


# -------------------------
# Download Inspection Report (PDF)
# -------------------------
@app.get("/download-report-pdf")
def download_report_pdf():
    global latest_results_df

    if latest_results_df is None or len(latest_results_df) == 0:
        return {"error": "No report available. Upload a file first."}

    df = latest_results_df.copy()

    keep_cols = [
        "consumer_id", "risk_score", "risk_band", "label", "action", "confidence",
        "layer1_physics", "layer2_rules", "layer3_collusion",
        "layer4_ml", "layer5_evidence", "layer6_human",
        "reason"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].sort_values("risk_score", ascending=False).head(20)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI Smart Meter Theft Risk - Inspection Report", styles["Title"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    total = len(latest_results_df)
    high = int((latest_results_df["risk_score"] >= 80).sum()) if "risk_score" in latest_results_df.columns else 0

    elements.append(Paragraph(f"Total Consumers Analyzed: {total}", styles["Normal"]))
    elements.append(Paragraph(f"High Risk (>=80): {high}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Top 20 High-Risk Consumers", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Inspector Notes: _________________________________", styles["Normal"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Signature: _____________________", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=inspection_report.pdf"},
    )

@app.get("/heatmap")
def area_heatmap(request: Request, db: Session = Depends(get_db)):

    rows = db.query(ConsumerRisk).all()
    if not rows:
        return templates.TemplateResponse("heatmap.html", {
            "request": request,
            "zones": []
        })

    # Fake zone mapping (for demo)
    def assign_zone(cid: str) -> str:
        try:
            num = int(cid.replace("C", ""))
        except:
            num = 0

        if num <= 20:
            return "Zone A"
        elif num <= 40:
            return "Zone B"
        elif num <= 60:
            return "Zone C"
        elif num <= 80:
            return "Zone D"
        return "Zone E"

    zone_scores = {}

    for r in rows:
        z = assign_zone(r.consumer_id)
        zone_scores.setdefault(z, []).append(float(r.risk_score))

    zones = []
    for z, scores in zone_scores.items():
        avg_score = sum(scores) / len(scores)
        zones.append({
            "zone": z,
            "avg_risk": round(avg_score, 2),
            "consumers": len(scores)
        })

    zones = sorted(zones, key=lambda x: x["avg_risk"], reverse=True)

    return templates.TemplateResponse("heatmap.html", {
        "request": request,
        "zones": zones
    })
