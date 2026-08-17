# System Architecture Documentation

## AI-Based Disease Prediction System

---

## 1. System Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  React Frontend (Vite)                       │   │
│  │                                                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │Auth Pages│  │Dashboard │  │Prediction│  │ History  │  │   │
│  │  │Login/Reg │  │Stats/    │  │Forms     │  │Models    │  │   │
│  │  │          │  │Charts    │  │Results   │  │Admin     │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  │                                                              │   │
│  │  Auth Context │ API Service (Axios) │ React Router          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │ HTTPS / REST
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       API LAYER (FastAPI)                            │
│                                                                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│  │  /auth  │ │/predict │ │/history │ │/dash-   │ │/models  │    │
│  │         │ │/heart   │ │         │ │board    │ │/perf    │    │
│  │register │ │/diabetes│ │  list   │ │  stats  │ │         │    │
│  │login    │ │/breast- │ │ detail  │ │         │ │/admin   │    │
│  │me       │ │ cancer  │ │ delete  │ │         │ │         │    │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘    │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              Middleware Layer                                │    │
│  │  JWT Auth Middleware │ CORS │ Error Handlers │ Validation   │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                    │                        │
          ┌─────────┘                        └─────────┐
          ▼                                            ▼
┌─────────────────────┐                   ┌──────────────────────┐
│    ML LAYER          │                   │    DATABASE LAYER     │
│                      │                   │                       │
│  MLPredictor         │                   │  MongoDB              │
│  (loads at startup)  │                   │                       │
│                      │                   │  ┌──────────────┐    │
│  ┌─────────────┐    │                   │  │    users     │    │
│  │heart model  │    │                   │  └──────────────┘    │
│  │(RF Pipeline)│    │                   │  ┌──────────────┐    │
│  └─────────────┘    │                   │  │ predictions  │    │
│  ┌─────────────┐    │                   │  └──────────────┘    │
│  │diabetes     │    │                   │  ┌──────────────┐    │
│  │(RF Pipeline)│    │                   │  │model_metrics │    │
│  └─────────────┘    │                   │  └──────────────┘    │
│  ┌─────────────┐    │                   └──────────────────────┘
│  │breast_cancer│    │
│  │(SVC Pipeline│    │
│  └─────────────┘    │
└─────────────────────┘
```

---

## 2. Data Flow Diagram (DFD)

### DFD Level 0 — Context Diagram

```
            ┌──────────────┐
            │   PATIENT /  │
            │     USER     │
            └──────┬───────┘
                   │  Medical Data Input
                   │  Login / Register
                   ▼
       ┌───────────────────────────┐
       │                           │
       │   AI DISEASE PREDICTION   │◄──── ML Models (trained offline)
       │         SYSTEM            │
       │                           │
       └───────────────────────────┘
                   │
                   │  Prediction Results
                   │  Risk Probability
                   │  History / Analytics
                   ▼
            ┌──────────────┐
            │   PATIENT /  │
            │     USER     │
            └──────────────┘
```

### DFD Level 1 — System Processes

```
USER
 │
 ├──[Register/Login]──► 1.0 Authentication
 │                           │
 │                           ├──[Validate]──► Users DB
 │                           └──[JWT Token]──► USER
 │
 ├──[Medical Data]──► 2.0 Input Validation
 │                         │
 │                         ├──[Valid Data]──► 3.0 Preprocessing
 │                         │                      │
 │                         │                      ├──[Scaled Data]──► 4.0 Prediction
 │                         │                                               │
 │                         │                                               ├──[Result]──► 5.0 Store
 │                         │                                               │                  │
 │                         │                                               │                  └──► Predictions DB
 │                         └──[Invalid]──► Error Response ──► USER
 │
 ├──[View History]──► 6.0 History Query ──► Predictions DB ──► USER
 │
 └──[View Dashboard]──► 7.0 Analytics ──► Predictions DB + Model Files ──► USER
```

---

## 3. Sequence Diagram — Prediction Flow

```
USER          Frontend       Backend         MLPredictor      MongoDB
 │                │              │                 │              │
 │─Fill Form─────►│              │                 │              │
 │                │─Validate─────►              │              │
 │                │─POST /predict/heart──────────►│              │
 │                │              │─get_current_user►             │
 │                │              │              ◄──JWT Verified──│
 │                │              │─predictor.predict(disease, data)
 │                │              │                 │              │
 │                │              │─────────────────►│              │
 │                │              │         load pipeline           │
 │                │              │         preprocess input        │
 │                │              │         model.predict()         │
 │                │              │         model.predict_proba()   │
 │                │              │◄────────(pred, prob, model_name)│
 │                │              │                 │              │
 │                │              │─save prediction─────────────────►│
 │                │              │◄──────────────────────────────(id)
 │                │              │                 │              │
 │                │◄─────────────────JSON Response─┤              │
 │◄───────────────│              │                 │              │
 │  Show Result   │              │                 │              │
```

---

## 4. ER Diagram (MongoDB Document Model)

```
┌─────────────────────────────────────────────────┐
│                    USERS                        │
│─────────────────────────────────────────────────│
│  _id          : ObjectId (PK)                   │
│  name         : String                          │
│  email        : String (Unique Index)           │
│  password_hash: String                          │
│  role         : Enum["user", "admin"]           │
│  created_at   : DateTime                        │
│  updated_at   : DateTime                        │
└─────────────────┬───────────────────────────────┘
                  │ 1
                  │
                  │ N
┌─────────────────────────────────────────────────┐
│                 PREDICTIONS                     │
│─────────────────────────────────────────────────│
│  _id          : ObjectId (PK)                   │
│  user_id      : String (FK → users._id, Index) │
│  disease      : Enum["heart","diabetes",        │
│                        "breast_cancer"] (Index) │
│  input_data   : Object (feature key-value pairs)│
│  prediction   : Int (0 or 1)                   │
│  probability  : Float (0.0 - 1.0)              │
│  model_used   : String                          │
│  created_at   : DateTime                        │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│               MODEL_METRICS                     │
│─────────────────────────────────────────────────│
│  _id          : ObjectId (PK)                   │
│  disease      : String (Compound Index)         │
│  algorithm    : String (Compound Index)         │
│  accuracy     : Float                           │
│  precision    : Float                           │
│  recall       : Float                           │
│  f1_score     : Float                           │
│  roc_auc      : Float                           │
│  created_at   : DateTime                        │
└─────────────────────────────────────────────────┘
```

---

## 5. Use Case Diagram

```
                    ┌────────────────────────────────────────────┐
                    │              Disease Prediction System      │
                    │                                            │
  ┌──────┐          │  ┌─────────────────────────────────┐      │
  │      │──Register─► │ Register / Login                 │      │
  │      │          │  └─────────────────────────────────┘      │
  │      │          │                                            │
  │      │──Login───►  ┌─────────────────────────────────┐      │
  │      │          │  │ View Dashboard                   │      │
  │      │          │  └─────────────────────────────────┘      │
  │ USER │          │                                            │
  │      │──Predict─►  ┌─────────────────────────────────┐      │
  │      │          │  │ Predict Heart Disease            │      │
  │      │          │  │ Predict Diabetes                 │      │
  │      │          │  │ Predict Breast Cancer            │      │
  │      │          │  └─────────────────────────────────┘      │
  │      │──History─►  ┌─────────────────────────────────┐      │
  │      │          │  │ View Prediction History           │      │
  │      │          │  │ Delete Prediction                │      │
  │      │          │  └─────────────────────────────────┘      │
  │      │──Models──►  ┌─────────────────────────────────┐      │
  └──────┘          │  │ Compare ML Models                │      │
                    │  └─────────────────────────────────┘      │
                    │                                            │
  ┌──────┐          │  ┌─────────────────────────────────┐      │
  │      │──Admin───►  │ View All Users                   │      │
  │ADMIN │          │  │ View Platform Statistics         │      │
  │      │          │  │ View All Predictions (global)    │      │
  └──────┘          │  └─────────────────────────────────┘      │
                    │                                            │
                    └────────────────────────────────────────────┘
```

---

## 6. ML Pipeline Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           ML Training (Offline)         │
                    │                                         │
                    │  train_heart.py / train_diabetes.py /  │
                    │  train_breast_cancer.py                 │
                    │                                         │
                    │  ┌─────────────────────────────────┐  │
                    │  │         sklearn Pipeline         │  │
                    │  │                                  │  │
                    │  │  ColumnTransformer               │  │
                    │  │    └─ SimpleImputer (median)     │  │
                    │  │    └─ StandardScaler             │  │
                    │  │                    │             │  │
                    │  │  Classifier        │             │  │
                    │  │    └─ LogReg / SVM / RF / XGB   │  │
                    │  │                                  │  │
                    │  └─────────────────────────────────┘  │
                    │                 │                       │
                    │     ┌──────────┴───────────┐          │
                    │     ▼                       ▼          │
                    │  best_model.joblib    all_metrics.json │
                    │  scaler.joblib        metadata.json    │
                    └─────────────────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │         FastAPI Startup                  │
                    │                                         │
                    │  MLPredictor.load_models()              │
                    │  Loads all 3 disease pipelines          │
                    │  into memory (loaded once)              │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │         Per-Request Prediction           │
                    │                                         │
                    │  1. Receive input_data dict             │
                    │  2. Build DataFrame with feature order  │
                    │  3. pipeline.predict(df) → int         │
                    │  4. pipeline.predict_proba(df) → float │
                    │  5. Return (pred, prob, model_name)    │
                    └─────────────────────────────────────────┘
```

---

## 7. Security Architecture

```
                    HTTP Request
                         │
                         ▼
              ┌──────────────────────┐
              │    CORS Middleware    │
              │ Whitelist origins    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Rate Limiting       │
              │  (via uvicorn)       │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  JWT Auth Middleware │
              │                      │
              │  1. Extract Bearer   │
              │  2. Decode JWT       │
              │  3. Verify signature │
              │  4. Check expiry     │
              │  5. Load user from DB│
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Pydantic Validation │
              │  Input sanitization  │
              │  Type checking       │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Role Authorization  │
              │  user / admin check  │
              └──────────────────────┘
```
