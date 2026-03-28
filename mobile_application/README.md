# TenantShield — Mobile Application

An Android building inspection app that helps NYC tenants file formal housing complaints. Powered by a multi-agent AI system (A2A) using Google Gemini, with Firebase for authentication and data persistence.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Kotlin, Jetpack Compose, Material 3 |
| **Backend Logic** | Java |
| **Camera** | CameraX 1.4.1 |
| **AI Agents** | Gemini Live API (via Cloud Run ADC), Gemini REST API |
| **Auth** | Firebase Auth (Google Sign-In) |
| **Database** | Cloud Firestore |
| **Storage** | Firebase Cloud Storage |
| **Proxy** | Cloud Run (Python FastAPI) |
| **Build** | AGP 9.1.0, Kotlin 2.2.10, Compose BOM 2025.12.00 |

**Min SDK:** 24 | **Target SDK:** 36

---

## Features

### Home
- Dashboard with hero gradient banner and "Start Inspection" button
- Emergency hotlines (Flood & Water Safety, HVAC, Grid Integrity)
- Stats cards: Priority Alerts, Efficiency Index, Daily Briefing

### Inspection Flow (A2A Agent Pipeline)
1. **Conversation Screen** — AI agent collects tenant's name, address, unit, and complaint via voice/text
2. **Camera View** — Live CameraX preview with scanning frame, flashlight toggle, multi-image capture
3. **Image Review** — Horizontal pager with selection, EXIF-corrected orientation
4. **AI Inspection** — Gemini multimodal analysis against NYC Housing Maintenance Code
5. **Result Explanation** — Agent explains findings via TTS, generates formal complaint
6. **Auto-save** — Results saved to Firestore, images uploaded to Cloud Storage

### Reports
- Scrollable list of past complaints from Firestore (title, address, date, hazard badge)
- Checkbox selection for batch PDF download
- Tap to view full complaint detail
- PDF generation on-device (Android PdfDocument API)
- Downloads to phone's Downloads folder via MediaStore

### Authentication
- Google Sign-In via Credential Manager API
- Home accessible without login; Inspect and Reports require login
- 1-second warning popup redirects to login if not authenticated
- Profile screen with editable address and inspection history

### Profile
- User avatar, display name, email from Google account
- Editable current address (saved to Firestore)
- Inspection history with hazard-level badges
- Sign out

---

## Architecture

### A2A Agent System

```
User ←→ VoiceService (STT/TTS) ←→ InteractingAgent ←→ Cloud Run Proxy ←→ Gemini Live API
                                         ↕                    ↓ (fallback)
                                   AgentOrchestrator      Direct REST API
                                    ↕              ↕
                            InspectionAgent    FilingAgent
                            (REST/Proxy)       (REST/Proxy)
```

### Pipeline States

```
IDLE → COLLECTING_INFO → AWAITING_IMAGES → INSPECTING → EXPLAINING_RESULTS → FILING → COMPLETE
```

### Agents

| Agent | Role | API |
|-------|------|-----|
| **Interacting Agent** | Collects user info via voice conversation, explains results | Cloud Run Proxy → Gemini Live API |
| **Inspection Agent** | Analyzes captured images for housing code violations | Gemini REST API (multimodal) |
| **Filing Agent** | Generates formal complaint form with HMC legal citations | Gemini REST API |
| **Orchestrator** | Java state machine coordinating all agents | Local (no API) |

### Agent Prompts

Prompts follow the web application's NYC Housing Maintenance Code inspection protocol:

- **16 hazard types** with real HMC statute codes (§27-2031, §27-2017.1, etc.)
- **"Observe first, classify last"** — agents describe what they see before classifying
- **Deterministic classification** — Class C (24hr), Class B (30 days), Class A (90 days)
- **Anti-hallucination** — agents never invent legal statutes, only use provided HMC codes
- **Specific verification questions** per hazard type before classification

---

## Cloud Run Proxy (ADC)

Located in `cloud-proxy/`. A lightweight FastAPI service deployed on Cloud Run that handles Gemini API authentication using Application Default Credentials.

**Endpoints:**
| Endpoint | Purpose |
|----------|---------|
| `POST /session/start` | Start a Live API conversation session |
| `POST /session/send` | Send a message in a session |
| `POST /session/end` | End a session |
| `POST /analyze` | Multimodal image analysis |
| `POST /generate` | Text generation |
| `GET /health` | Health check |

**Fallback:** If the proxy is unavailable, the app falls back to direct REST API calls with a Google AI Studio API key.

---

## Firebase Integration

### Firestore Structure
```
users/{userId}/
  ├── tenant_name, address, unit_number, updated_at
  └── inspections/{autoId}/
        ├── hazard_level, overall_severity, raw_analysis
        ├── findings[], recommended_actions[], image_urls[]
        ├── document_id, filing_date, tenant_name, address
        ├── nature_of_complaint, hazard_class, inspector_signature
        └── created_at
```

### Cloud Storage Structure
```
inspections/{userId}/{inspectionId}/photo_1.jpg, photo_2.jpg, ...
```

### Security Rules
- Users can only read/write their own data (`request.auth.uid == userId`)

---

## Project Structure

```
mobile_application/
├── app/src/main/java/com/example/tenantshield/
│   ├── MainActivity.kt                    # Navigation, auth gating, top/bottom bars
│   ├── agents/
│   │   ├── config/GeminiConfig.java       # Cloud Run proxy + REST API config
│   │   ├── models/                        # UserInfo, InspectionResult, ComplaintForm, etc.
│   │   ├── orchestrator/                  # AgentOrchestrator state machine
│   │   ├── service/
│   │   │   ├── GeminiApiService.java      # REST API wrapper (OkHttp)
│   │   │   ├── GeminiLiveApiService.java  # WebSocket client (legacy)
│   │   │   ├── VertexAuthService.java     # Service account auth (legacy)
│   │   │   └── VoiceService.java          # Android STT + TTS
│   │   ├── interacting/InteractingAgent.java  # Cloud Run proxy conversation
│   │   ├── inspection/InspectionAgent.java    # Multimodal image analysis
│   │   ├── filing/FilingAgent.java            # Complaint form generation
│   │   └── firebase/
│   │       ├── FirebaseAuthService.java   # Google Sign-In, email auth
│   │       ├── FirestoreService.java      # User profiles, inspection CRUD
│   │       └── StorageService.java        # Image uploads
│   └── ui/
│       ├── viewmodel/InspectionViewModel.kt  # StateFlow bridge
│       ├── screens/
│       │   ├── auth/LoginScreen.kt        # Google Sign-In
│       │   ├── home/HomeScreen.kt         # Dashboard
│       │   ├── inspect/InspectScreen.kt   # Camera + conversation
│       │   ├── owners/OwnersScreen.kt     # Owner reveal
│       │   ├── profile/ProfileScreen.kt   # User profile + history
│       │   └── reports/ReportsScreen.kt   # Report list + detail + PDF
│       └── theme/                         # Color, Type, Theme
├── cloud-proxy/
│   ├── main.py                            # FastAPI proxy service
│   ├── requirements.txt
│   └── Dockerfile
└── README.md
```

---

## Setup

### Prerequisites
- Android Studio (latest, supporting AGP 9.1.0)
- Google Cloud project with billing enabled
- Firebase project linked to GCP

### 1. Clone and Open
```bash
git clone <repo-url>
# Open mobile_application/ in Android Studio (not the monorepo root)
```

### 2. Create `local.properties`
```properties
sdk.dir=/path/to/Android/sdk
GEMINI_API_KEY=your_google_ai_studio_api_key
FIREBASE_WEB_CLIENT_ID=your_firebase_web_client_id.apps.googleusercontent.com
```

### 3. Add Firebase Config
Place `google-services.json` (from Firebase Console) in:
```
app/google-services.json
```

### 4. (Optional) Service Account for Direct Vertex AI
Place service account JSON in:
```
app/src/main/assets/service-account.json
```
Not needed if using Cloud Run proxy.

### 5. Deploy Cloud Run Proxy
```bash
cd cloud-proxy
gcloud run deploy tenantshield-proxy \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=your-project-id,GOOGLE_CLOUD_REGION=us-central1
```

Grant Vertex AI permission to Cloud Run:
```bash
gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### 6. Build and Run
- Connect Android device or start emulator
- Click Run in Android Studio

---

## Security

All sensitive files are gitignored:
- `local.properties` — API keys
- `app/google-services.json` — Firebase config
- `app/src/main/assets/service-account.json` — GCP credentials
- `.env` — Environment variables

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| CameraX | 1.4.1 | Camera preview + image capture |
| OkHttp | 4.12.0 | HTTP client for API calls |
| Gson | 2.11.0 | JSON serialization |
| Google Auth Library | 1.23.0 | OAuth2 token generation |
| Firebase BOM | 33.7.0 | Auth, Firestore, Cloud Storage |
| Credentials API | 1.3.0 | Google Sign-In (Credential Manager) |
| Google ID | 1.1.1 | Google ID token extraction |
| ExifInterface | 1.3.7 | Image rotation correction |
| Material Icons Extended | BOM | Full Material icon set |
| ViewModel Compose | 2.8.7 | Compose + ViewModel integration |

### Permissions
- `CAMERA` — Photo capture
- `INTERNET` — API calls
- `RECORD_AUDIO` — Voice interaction (STT)
