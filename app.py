from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, Query
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import pandas as pd
import os

from database.db import engine, Base
from database.deps import get_db
from models.models import Upload, UsageRecord, ConsumerRisk, Feedback

from services.preprocess import preprocess_data
from services.features import make_features
from services.riskmodel import run_isolation_forest
from services.explain import generate_reason, recommend_action

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Smart Meter Theft Risk System")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv", ".txt"}



def allowed_file(filename: str):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


def read_any_file(filepath: str):
    filepath_lower = filepath.lower()

    if filepath_lower.endswith(".csv"):
        return pd.read_csv(filepath)

    if filepath_lower.endswith(".tsv"):
        return pd.read_csv(filepath, sep="\t")

    if filepath_lower.endswith(".xlsx"):
        return pd.read_excel(filepath, engine="openpyxl")

    if filepath_lower.endswith(".xls"):
        return pd.read_excel(filepath, engine="xlrd")

    raise ValueError("Unsupported file format")


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
    if not allowed_file(file.filename):
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "msg": " Unsupported file type! Upload CSV/XLSX/XLS/TSV only."
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

    # Store usage records into DB
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

    # Run model
    results = run_isolation_forest(features)

    # Store consumer risk into DB
    db.query(ConsumerRisk).delete()
    db.commit()

    for _, r in results.iterrows():
        reason = generate_reason(r)
        action = recommend_action(float(r["risk_score"]))

        db.add(ConsumerRisk(
            consumer_id=str(r["consumer_id"]),
            risk_score=float(r["risk_score"]),
            label=str(r["label"]),
            reason=reason,
            action=action
        ))
    db.commit()

    preview = df.head(5).to_dict(orient="records")

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "msg": " File uploaded, processed, and stored in DB successfully!",
        "preview": preview
    })


# -------------------------
# Dashboard
# -------------------------
@app.get("/dashboard")
def dashboard(request: Request, db: Session = Depends(get_db)):
    total_consumers = db.query(ConsumerRisk).count()
    total_records = db.query(UsageRecord).count()
    anomalies = db.query(ConsumerRisk).filter(ConsumerRisk.label == "Anomaly").count()
    critical = db.query(ConsumerRisk).filter(ConsumerRisk.risk_score >= 80).count()

    top_risky = (
        db.query(ConsumerRisk)
        .order_by(ConsumerRisk.risk_score.desc())
        .limit(5)
        .all()
    )

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "total_consumers": total_consumers,
        "total_records": total_records,
        "anomalies": anomalies,
        "critical": critical,
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

    if filter == "critical":
        query = query.filter(ConsumerRisk.risk_score >= 80)

    elif filter == "anomaly":
        query = query.filter(ConsumerRisk.label == "Anomaly")

    elif filter == "normal":
        query = query.filter(ConsumerRisk.label == "Normal")

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
# Review Page
# -------------------------
@app.get("/review")
def review_page(request: Request):
    return templates.TemplateResponse("review.html", {"request": request})


@app.post("/review")
def submit_review(
    consumer_id: str = Form(...),
    decision: str = Form(...),
    note: str = Form(None),
    db: Session = Depends(get_db)
):
    db.add(Feedback(
        consumer_id=consumer_id,
        decision=decision,
        note=note
    ))
    db.commit()

    return RedirectResponse(url="/alerts", status_code=303)
