# app.py - Firebase Version
import os
import json
import hashlib
import secrets
import datetime
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
firebase_creds = json.loads(os.getenv("FIREBASE_CREDENTIALS"))
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

# App setup
app = FastAPI(title="Citizen AI", description="AI-powered citizen services platform")

# Create directories if they don't exist
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Fallback responses
FALLBACK_RESPONSES = [
    "I understand your query about Indian government services. For the most accurate and up-to-date information, I recommend visiting the official government portal at india.gov.in or contacting your nearest government office.",
    # ... (other fallback responses)
]

class FirebaseDatabase:
    def _hash_password(self, password: str, salt: Optional[str] = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt.encode('utf-8'), 
            100000
        )
        return password_hash.hex(), salt
    
    def register_user(self, username: str, email: str, password: str, full_name: str) -> Dict:
        """Register a new user"""
        if len(password) < 8:
            return {"success": False, "message": "Password must be at least 8 characters"}
        
        password_hash, salt = self._hash_password(password)
        
        try:
            user_ref = db.collection("users").document(username)
            if user_ref.get().exists:
                return {"success": False, "message": "Username already exists"}
            
            # Check if email exists
            email_query = db.collection("users").where("email", "==", email).limit(1).get()
            if len(email_query) > 0:
                return {"success": False, "message": "Email already exists"}
            
            user_ref.set({
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "salt": salt,
                "full_name": full_name,
                "created_at": datetime.datetime.now().isoformat(),
                "last_login": None,
                "is_active": True
            })
            
            return {"success": True, "message": "Registration successful", "user_id": username}
        
        except Exception as e:
            return {"success": False, "message": f"Registration failed: {str(e)}"}
    
    def login_user(self, username: str, password: str) -> Dict:
        """Authenticate user and create session"""
        try:
            # Try username first
            user_ref = db.collection("users").document(username)
            user_doc = user_ref.get()
            
            # If not found by username, try email
            if not user_doc.exists:
                email_query = db.collection("users").where("email", "==", username).limit(1).get()
                if len(email_query) == 0:
                    return {"success": False, "message": "Invalid credentials"}
                
                user_doc = email_query[0]
                username = user_doc.id
            
            user_data = user_doc.to_dict()
            
            if not user_data.get("is_active", True):
                return {"success": False, "message": "Account is inactive"}
            
            # Verify password
            input_hash, _ = self._hash_password(password, user_data["salt"])
            if input_hash == user_data["password_hash"]:
                session_token = secrets.token_urlsafe(32)
                expires_at = datetime.datetime.now() + datetime.timedelta(days=7)
                
                # Create session
                db.collection("sessions").document(session_token).set({
                    "user_id": username,
                    "expires_at": expires_at.isoformat(),
                    "is_active": True
                })
                
                # Update last login
                user_ref.update({"last_login": datetime.datetime.now().isoformat()})
                
                return {
                    "success": True,
                    "session_token": session_token,
                    "user_id": username,
                    "username": username,
                    "full_name": user_data["full_name"]
                }
            else:
                return {"success": False, "message": "Invalid credentials"}
        
        except Exception as e:
            return {"success": False, "message": f"Login failed: {str(e)}"}
    
    def verify_session(self, session_token: str) -> Optional[Dict]:
        """Verify session token"""
        if not session_token:
            return None
            
        try:
            session_ref = db.collection("sessions").document(session_token)
            session_doc = session_ref.get()
            
            if not session_doc.exists:
                return None
                
            session_data = session_doc.to_dict()
            expires_at = datetime.datetime.fromisoformat(session_data["expires_at"])
            
            if expires_at < datetime.datetime.now() or not session_data.get("is_active", True):
                return None
                
            # Get user data
            user_ref = db.collection("users").document(session_data["user_id"])
            user_doc = user_ref.get()
            
            if not user_doc.exists or not user_doc.to_dict().get("is_active", True):
                return None
                
            user_data = user_doc.to_dict()
            
            return {
                "user_id": session_data["user_id"],
                "username": session_data["user_id"],
                "full_name": user_data["full_name"]
            }
        except Exception:
            return None
    
    def save_chat(self, user_id: str, user_message: str, ai_response: str):
        """Save chat interaction"""
        db.collection("chat_history").add({
            "user_id": user_id,
            "user_message": user_message,
            "ai_response": ai_response,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def save_sentiment(self, user_id: str, feedback_text: str, sentiment: str, confidence: float = 0.0):
        """Save sentiment analysis result"""
        db.collection("sentiment_analysis").add({
            "user_id": user_id,
            "feedback_text": feedback_text,
            "sentiment": sentiment,
            "confidence": confidence,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def get_chat_history(self, user_id: str, limit: int = 10):
        """Get user's chat history"""
        docs = db.collection("chat_history")\
                .where("user_id", "==", user_id)\
                .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .stream()
        
        return [{
            "user_message": doc.get("user_message"),
            "ai_response": doc.get("ai_response"),
            "timestamp": doc.get("timestamp")
        } for doc in docs]
    
    def get_sentiment_stats(self):
        """Get sentiment analysis statistics"""
        docs = db.collection("sentiment_analysis").stream()
        
        stats = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for doc in docs:
            sentiment = doc.get("sentiment")
            if sentiment in stats:
                stats[sentiment] += 1
                
        return stats

# Initialize database
db = FirebaseDatabase()

# IBM Watsonx configuration
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

class SimpleCitizenAI:
    def __init__(self):
        self.prompt_template = self.create_prompt_template()
        self.llm = WatsonxLLM(
            model_id=os.getenv("WATSONX_MODEL_ID"),
            url=WATSONX_URL,
            apikey=WATSONX_APIKEY,
            project_id=WATSONX_PROJECT_ID,
            params={
                "decoding_method": "greedy",
                "max_new_tokens": 500,
                "temperature": 0.7
            }
        )

    def create_prompt_template(self):
        return PromptTemplate(
            input_variables=["user_question"],
            template="""You are Citizen AI, a smart assistant for Indian citizens. You help with questions about:
- Indian government services and schemes
- Sustainable city planning following Indian regulations
- Energy efficiency and renewable energy in Indian context
- Water management following Indian policies
- Waste management as per Indian guidelines
- Air quality and pollution control (Indian standards)
- Smart transportation systems in India
- Green building practices (Indian Green Building Council standards)
- Environmental monitoring per Indian regulations

IMPORTANT GUIDELINES:
- All responses must comply with Indian laws and the Indian Constitution
- For sensitive legal, financial, or personal matters, advise: "Please contact your nearest government office or authorized service center for detailed assistance"
- Provide practical solutions relevant to Indian cities and regulations
- Reference Indian government schemes and initiatives when applicable

User Question: {user_question}

Provide a helpful, actionable response focused on Indian context and regulations:"""
        )

    def generate_response(self, user_message: str) -> str:
        """Generate AI response using IBM Watsonx LLM"""
        try:
            prompt = self.prompt_template.format(user_question=user_message)
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Sorry, there was an error connecting to the AI service: {str(e)}"

# Initialize AI
ai = SimpleCitizenAI()

def get_current_user(session_token: Optional[str] = Cookie(None)):
    """Get current user from session"""
    if not session_token:
        return None
    return db.verify_session(session_token)

def analyse_sentiment(text: str):
    """Simple sentiment analysis using keywords"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                     'satisfied', 'happy', 'pleased', 'impressed', 'helpful', 'efficient']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointed', 'frustrated',
                     'angry', 'unsatisfied', 'poor', 'waste', 'useless', 'slow']
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    if positive_count > negative_count:
        sentiment = "Positive"
        confidence = min(0.6 + (positive_count - negative_count) * 0.1, 1.0)
    elif negative_count > positive_count:
        sentiment = "Negative"
        confidence = min(0.6 + (negative_count - positive_count) * 0.1, 1.0)
    else:
        sentiment = "Neutral"
        confidence = 0.5
    return sentiment, confidence

# Routes (remain exactly the same as in your original code)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, session_token: Optional[str] = Cookie(None)):
    user = get_current_user(session_token)
    if user:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(...)
):
    result = db.register_user(username, email, password, full_name)
    
    if result["success"]:
        return RedirectResponse(url="/login?message=Registration successful", status_code=302)
    else:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": result["message"]
        })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, message: Optional[str] = None):
    return templates.TemplateResponse("login.html", {"request": request, "message": message})

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    result = db.login_user(username, password)
    
    if result["success"]:
        response = RedirectResponse(url="/dashboard", status_code=302)
        response.set_cookie(key="session_token", value=result["session_token"], httponly=True)
        return response
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": result["message"]
        })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, session_token: Optional[str] = Cookie(None)):
    user = get_current_user(session_token)
    if not user:
        return RedirectResponse(url="/login")
    
    # Get recent chat history
    chat_history = db.get_chat_history(user["user_id"], 5)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "chat_history": chat_history
    })

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, session_token: Optional[str] = Cookie(None)):
    user = get_current_user(session_token)
    if not user:
        return RedirectResponse(url="/login")
    
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})

@app.post("/chat")
async def chat(
    request: Request,
    message: str = Form(...),
    session_token: Optional[str] = Cookie(None)
):
    user = get_current_user(session_token)
    if not user:
        return RedirectResponse(url="/login")
    
    # Generate AI response
    ai_response = ai.generate_response(message)
    
    # Save chat to database
    db.save_chat(user["user_id"], message, ai_response)
    
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "user": user,
        "user_message": message,
        "ai_response": ai_response
    })

@app.get("/feedback", response_class=HTMLResponse)
async def feedback_page(request: Request, session_token: Optional[str] = Cookie(None)):
    user = get_current_user(session_token)
    if not user:
        return RedirectResponse(url="/login")
    
    return templates.TemplateResponse("feedback.html", {"request": request, "user": user})

@app.post("/feedback")
async def submit_feedback(
    request: Request,
    feedback: str = Form(...),
    session_token: Optional[str] = Cookie(None)
):
    user = get_current_user(session_token)
    if not user:
        return RedirectResponse(url="/login")
    
    # Analyze sentiment
    sentiment, confidence = analyse_sentiment(feedback)
    
    # Save to database
    db.save_sentiment(user["user_id"], feedback, sentiment, confidence)
    
    return templates.TemplateResponse("feedback.html", {
        "request": request,
        "user": user,
        "success": "Thank you for your feedback! Your sentiment has been analyzed and recorded.",
        "sentiment": sentiment
    })

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, session_token: Optional[str] = Cookie(None)):
    user = get_current_user(session_token)
    if not user:
        return RedirectResponse(url="/login")
    
    # Get sentiment statistics
    sentiment_stats = db.get_sentiment_stats()
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "user": user,
        "sentiment_stats": sentiment_stats
    })

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(key="session_token")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)