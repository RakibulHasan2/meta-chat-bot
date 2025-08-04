from fastapi import FastAPI, APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
import hmac
import hashlib
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import requests
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="Facebook Auto-Comment Reply System")
api_router = APIRouter(prefix="/api")

# Global AI Models
paraphrase_model = None
paraphrase_tokenizer = None
sentiment_analyzer = None

# Load AI Models on startup
@app.on_event("startup")
async def load_models():
    global paraphrase_model, paraphrase_tokenizer, sentiment_analyzer
    try:
        # Load T5 for paraphrasing
        model_name = os.environ.get('PARAPHRASE_MODEL', 't5-small')
        paraphrase_tokenizer = T5Tokenizer.from_pretrained(model_name)
        paraphrase_model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Load sentiment analyzer
        sentiment_model = os.environ.get('SENTIMENT_MODEL', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
        sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
        
        logger.info("AI Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading AI models: {e}")

# Models
class CommentData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    post_id: str
    comment_id: str
    page_id: str
    comment_text: str
    author_name: str
    author_id: str
    classification: Optional[str] = None
    sentiment: Optional[str] = None
    reply_text: Optional[str] = None
    replied: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PageConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_id: str
    page_name: str
    access_token: str
    active: bool = True
    response_templates: Dict[str, List[str]] = Field(default_factory=dict)
    auto_reply_enabled: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ParaphraseRequest(BaseModel):
    text: str
    num_paraphrases: int = 2

class ReplyRequest(BaseModel):
    comment_id: str
    reply_text: str

# Facebook Configuration
APP_SECRET = os.environ.get('FB_APP_SECRET', 'demo_app_secret_abcdef')
VERIFY_TOKEN = os.environ.get('FB_VERIFY_TOKEN', 'demo_verify_token_xyz')

# Response Templates
DEFAULT_TEMPLATES = {
    "greeting": [
        "Thank you for your comment! We appreciate your engagement.",
        "Hello! Thanks for reaching out to us.",
        "Hi there! We're glad you're part of our community."
    ],
    "question": [
        "Thank you for your question! We'll get back to you soon.",
        "Great question! Our team will provide you with more details.",
        "Thanks for asking! We're here to help you."
    ],
    "positive": [
        "Thank you so much for your kind words! We really appreciate it.",
        "We're thrilled to hear you're happy! Thank you for sharing.",
        "Your positive feedback means the world to us! Thank you."
    ],
    "negative": [
        "We appreciate your feedback and take all concerns seriously.",
        "Thank you for bringing this to our attention. We'll look into it.",
        "We understand your concern and want to make things right."
    ],
    "general": [
        "Thank you for your comment! We value your engagement.",
        "Thanks for being part of our community!",
        "We appreciate you taking the time to comment."
    ]
}

# Utility Functions
def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify Facebook webhook signature"""
    if not signature.startswith('sha1='):
        return False
    
    expected_signature = hmac.new(
        APP_SECRET.encode('utf-8'),
        payload,
        hashlib.sha1
    ).hexdigest()
    
    return hmac.compare_digest(signature[5:], expected_signature)

async def classify_comment(text: str) -> tuple:
    """Classify comment intent and sentiment"""
    try:
        # Get sentiment
        sentiment_result = sentiment_analyzer(text)[0]
        sentiment = sentiment_result['label'].lower()
        
        # Simple rule-based classification
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
            classification = 'greeting'
        elif any(word in text_lower for word in ['?', 'how', 'what', 'when', 'where', 'why', 'can you']):
            classification = 'question'
        elif sentiment in ['positive']:
            classification = 'positive'
        elif sentiment in ['negative']:
            classification = 'negative'
        else:
            classification = 'general'
            
        return classification, sentiment
    except Exception as e:
        logger.error(f"Error classifying comment: {e}")
        return 'general', 'neutral'

async def paraphrase_text(text: str, num_paraphrases: int = 2) -> List[str]:
    """Generate paraphrased versions of text using T5"""
    try:
        input_text = f"paraphrase: {text}"
        input_ids = paraphrase_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        paraphrases = []
        for i in range(num_paraphrases):
            with torch.no_grad():
                outputs = paraphrase_model.generate(
                    input_ids,
                    max_length=100,
                    num_beams=4,
                    temperature=0.7 + (i * 0.1),  # Vary temperature for diversity
                    do_sample=True,
                    early_stopping=True
                )
            
            paraphrased = paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if paraphrased not in paraphrases and paraphrased != text:
                paraphrases.append(paraphrased)
        
        return paraphrases if paraphrases else [text]
    except Exception as e:
        logger.error(f"Error paraphrasing text: {e}")
        return [text]

async def get_page_config(page_id: str) -> Optional[PageConfig]:
    """Get page configuration from database"""
    try:
        config_data = await db.page_configs.find_one({"page_id": page_id})
        if config_data:
            return PageConfig(**config_data)
        return None
    except Exception as e:
        logger.error(f"Error getting page config: {e}")
        return None

async def post_facebook_reply(page_id: str, comment_id: str, message: str) -> bool:
    """Post reply to Facebook comment"""
    try:
        page_config = await get_page_config(page_id)
        if not page_config:
            logger.error(f"No configuration found for page {page_id}")
            return False
        
        url = f"https://graph.facebook.com/v18.0/{comment_id}/comments"
        data = {
            'message': message,
            'access_token': page_config.access_token
        }
        
        # For demo mode, just log the action
        if page_config.access_token.startswith('demo_'):
            logger.info(f"DEMO MODE: Would reply to comment {comment_id} with: {message}")
            return True
        
        response = requests.post(url, data=data)
        if response.status_code == 200:
            logger.info(f"Successfully replied to comment {comment_id}")
            return True
        else:
            logger.error(f"Failed to reply to comment: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error posting Facebook reply: {e}")
        return False

async def process_comment(comment_data: dict, page_id: str):
    """Process incoming comment and generate reply"""
    try:
        # Extract comment information
        comment_id = comment_data.get('id')
        message = comment_data.get('message', '')
        author_name = comment_data.get('from', {}).get('name', 'Unknown')
        author_id = comment_data.get('from', {}).get('id', '')
        post_id = comment_data.get('post_id', '')
        
        if not message:
            return
        
        # Classify comment
        classification, sentiment = await classify_comment(message)
        
        # Get page config
        page_config = await get_page_config(page_id)
        if not page_config or not page_config.auto_reply_enabled:
            return
        
        # Select appropriate template
        templates = page_config.response_templates.get(classification, DEFAULT_TEMPLATES.get(classification, DEFAULT_TEMPLATES['general']))
        base_reply = templates[0]  # Use first template
        
        # Paraphrase the reply
        paraphrases = await paraphrase_text(base_reply, 1)
        final_reply = paraphrases[0] if paraphrases else base_reply
        
        # Save comment data
        comment_obj = CommentData(
            post_id=post_id,
            comment_id=comment_id,
            page_id=page_id,
            comment_text=message,
            author_name=author_name,
            author_id=author_id,
            classification=classification,
            sentiment=sentiment,
            reply_text=final_reply
        )
        
        await db.comments.insert_one(comment_obj.dict())
        
        # Post reply
        success = await post_facebook_reply(page_id, comment_id, final_reply)
        if success:
            await db.comments.update_one(
                {"comment_id": comment_id},
                {"$set": {"replied": True}}
            )
        
        logger.info(f"Processed comment from {author_name}: {classification} -> {final_reply}")
        
    except Exception as e:
        logger.error(f"Error processing comment: {e}")

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Facebook Auto-Comment Reply System", "status": "running"}

# Facebook Webhook Verification
@api_router.get("/webhook")
async def verify_webhook(request: Request):
    """Facebook webhook verification"""
    mode = request.query_params.get('hub.mode')
    token = request.query_params.get('hub.verify_token')
    challenge = request.query_params.get('hub.challenge')
    
    if mode == 'subscribe' and token == VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        return PlainTextResponse(challenge)
    else:
        logger.error("Webhook verification failed")
        raise HTTPException(status_code=403, detail="Forbidden")

# Facebook Webhook Handler
@api_router.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Facebook webhook events"""
    try:
        body = await request.body()
        signature = request.headers.get('X-Hub-Signature', '')
        
        # Verify signature (skip in demo mode)
        if not APP_SECRET.startswith('demo_') and not verify_webhook_signature(body, signature):
            raise HTTPException(status_code=403, detail="Invalid signature")
        
        data = json.loads(body)
        
        # Process webhook data
        for entry in data.get('entry', []):
            page_id = entry.get('id')
            
            # Handle page comments
            for change in entry.get('changes', []):
                if change.get('field') == 'feed':
                    value = change.get('value', {})
                    if value.get('item') == 'comment' and value.get('verb') == 'add':
                        comment_data = value.get('comment_id')
                        if comment_data:
                            # Get full comment data (in real implementation, you'd fetch from Facebook)
                            # For demo, create mock data
                            mock_comment = {
                                'id': value.get('comment_id'),
                                'message': f"Demo comment text for testing",
                                'from': {
                                    'name': 'Demo User',
                                    'id': 'demo_user_123'
                                },
                                'post_id': value.get('post_id', 'demo_post_123')
                            }
                            background_tasks.add_task(process_comment, mock_comment, page_id)
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error handling webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Paraphrasing API
@api_router.post("/paraphrase")
async def paraphrase_endpoint(request: ParaphraseRequest):
    """Paraphrase text using AI"""
    try:
        paraphrases = await paraphrase_text(request.text, request.num_paraphrases)
        return {"paraphrases": paraphrases}
    except Exception as e:
        logger.error(f"Error in paraphrase endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error generating paraphrases")

# Page Management
@api_router.post("/pages")
async def add_page(page_config: PageConfig):
    """Add or update page configuration"""
    try:
        # Set default templates if not provided
        if not page_config.response_templates:
            page_config.response_templates = DEFAULT_TEMPLATES
        
        await db.page_configs.update_one(
            {"page_id": page_config.page_id},
            {"$set": page_config.dict()},
            upsert=True
        )
        return {"status": "success", "message": "Page configuration saved"}
    except Exception as e:
        logger.error(f"Error saving page config: {e}")
        raise HTTPException(status_code=500, detail="Error saving page configuration")

@api_router.get("/pages", response_model=List[PageConfig])
async def get_pages():
    """Get all page configurations"""
    try:
        pages = await db.page_configs.find().to_list(1000)
        return [PageConfig(**page) for page in pages]
    except Exception as e:
        logger.error(f"Error getting pages: {e}")
        raise HTTPException(status_code=500, detail="Error fetching pages")

# Comments and Activity
@api_router.get("/comments", response_model=List[CommentData])
async def get_comments(limit: int = 50):
    """Get recent comments"""
    try:
        comments = await db.comments.find().sort("timestamp", -1).limit(limit).to_list(limit)
        return [CommentData(**comment) for comment in comments]
    except Exception as e:
        logger.error(f"Error getting comments: {e}")
        raise HTTPException(status_code=500, detail="Error fetching comments")

@api_router.post("/reply")
async def manual_reply(request: ReplyRequest):
    """Manually reply to a comment"""
    try:
        # Get comment data
        comment = await db.comments.find_one({"comment_id": request.comment_id})
        if not comment:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        # Paraphrase the reply
        paraphrases = await paraphrase_text(request.reply_text, 1)
        final_reply = paraphrases[0] if paraphrases else request.reply_text
        
        # Post reply
        success = await post_facebook_reply(comment['page_id'], request.comment_id, final_reply)
        
        if success:
            await db.comments.update_one(
                {"comment_id": request.comment_id},
                {"$set": {"replied": True, "reply_text": final_reply}}
            )
            return {"status": "success", "reply": final_reply}
        else:
            raise HTTPException(status_code=500, detail="Failed to post reply")
            
    except Exception as e:
        logger.error(f"Error in manual reply: {e}")
        raise HTTPException(status_code=500, detail="Error posting reply")

# Demo endpoint to test the system
@api_router.post("/demo/comment")
async def demo_comment(request: dict):
    """Demo endpoint to test comment processing"""
    try:
        comment_text = request.get('comment_text', '')
        page_id = request.get('page_id', 'demo_page_id_456')
        
        mock_comment = {
            'id': f'demo_comment_{uuid.uuid4()}',
            'message': comment_text,
            'from': {
                'name': 'Demo User',
                'id': 'demo_user_123'
            },
            'post_id': 'demo_post_123'
        }
        
        await process_comment(mock_comment, page_id)
        return {"status": "success", "message": "Demo comment processed"}
        
    except Exception as e:
        logger.error(f"Error in demo comment: {e}")
        raise HTTPException(status_code=500, detail="Error processing demo comment")

# Include router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()