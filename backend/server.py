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
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
import requests
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="Context-Aware Facebook Auto-Reply System")
api_router = APIRouter(prefix="/api")

# Global AI Models
paraphrase_model = None
paraphrase_tokenizer = None
sentiment_analyzer = None
embedding_model = None
knowledge_bases = {}  # Page-specific knowledge bases

# Load AI Models on startup
@app.on_event("startup")
async def load_models():
    global paraphrase_model, paraphrase_tokenizer, sentiment_analyzer, embedding_model
    try:
        # Load T5 for paraphrasing
        model_name = os.environ.get('PARAPHRASE_MODEL', 't5-small')
        paraphrase_tokenizer = T5Tokenizer.from_pretrained(model_name)
        paraphrase_model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Load sentiment analyzer
        sentiment_model = os.environ.get('SENTIMENT_MODEL', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
        sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
        
        # Load embedding model for context understanding
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load page-specific knowledge bases
        await load_page_knowledge_bases()
        
        logger.info("AI Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading AI models: {e}")

# Enhanced Models
class ProductInfo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_id: str
    name: str
    price: Optional[float] = None
    description: str
    category: str
    keywords: List[str] = Field(default_factory=list)
    availability: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PageKnowledge(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_id: str
    page_name: str
    business_type: str  # electronics, restaurant, clothing, etc.
    products: List[ProductInfo] = Field(default_factory=list)
    faqs: Dict[str, str] = Field(default_factory=dict)
    business_hours: Optional[str] = None
    location: Optional[str] = None
    contact_info: Optional[str] = None
    custom_responses: Dict[str, List[str]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PostData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_id: str
    post_id: str
    content: str
    post_type: str = "feed"  # feed, photo, video, etc.
    engagement_metrics: Dict[str, int] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

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
    intent: Optional[str] = None  # price_inquiry, product_info, complaint, etc.
    context_match: Optional[Dict] = None  # matched products/info
    reply_text: Optional[str] = None
    replied: bool = False
    confidence_score: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PageConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_id: str
    page_name: str
    business_type: str
    access_token: str
    active: bool = True
    auto_reply_enabled: bool = True
    auto_learning_enabled: bool = True
    confidence_threshold: float = 0.7
    response_templates: Dict[str, List[str]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Facebook Configuration
APP_SECRET = os.environ.get('FB_APP_SECRET', 'demo_app_secret_abcdef')
VERIFY_TOKEN = os.environ.get('FB_VERIFY_TOKEN', 'demo_verify_token_xyz')

# Intent classification keywords
INTENT_KEYWORDS = {
    'price_inquiry': ['price', 'cost', 'how much', 'expensive', 'cheap', 'rate', 'pricing', '$'],
    'product_info': ['specifications', 'details', 'features', 'specs', 'information', 'tell me about'],
    'availability': ['available', 'in stock', 'out of stock', 'when available', 'delivery'],
    'location': ['where', 'address', 'location', 'directions', 'how to reach'],
    'hours': ['open', 'close', 'hours', 'timing', 'when open', 'business hours'],
    'contact': ['phone', 'email', 'contact', 'call', 'reach', 'number'],
    'complaint': ['bad', 'terrible', 'worst', 'problem', 'issue', 'disappointed', 'angry'],
    'compliment': ['great', 'awesome', 'love', 'excellent', 'amazing', 'best', 'fantastic']
}

class ContextAwareAI:
    def __init__(self):
        self.page_embeddings = {}
        self.product_embeddings = {}
    
    async def analyze_comment_context(self, comment_text: str, page_id: str) -> Dict:
        """Analyze comment for intent and find relevant context"""
        try:
            # Get comment embedding
            comment_embedding = embedding_model.encode([comment_text])
            
            # Classify intent
            intent = self.classify_intent(comment_text)
            
            # Find relevant products/info for this page
            relevant_context = await self.find_relevant_context(
                comment_text, comment_embedding, page_id, intent
            )
            
            return {
                'intent': intent,
                'context': relevant_context,
                'confidence': relevant_context.get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing comment context: {e}")
            return {'intent': 'general', 'context': {}, 'confidence': 0.0}
    
    def classify_intent(self, text: str) -> str:
        """Classify the intent of the comment"""
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, keywords in INTENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        return 'general'
    
    async def find_relevant_context(self, comment_text: str, comment_embedding, page_id: str, intent: str) -> Dict:
        """Find relevant products/information for the comment"""
        try:
            # Get page knowledge
            page_knowledge = knowledge_bases.get(page_id)
            if not page_knowledge:
                return {'confidence': 0.0}
            
            relevant_info = {'confidence': 0.0}
            
            # Search products if intent is product-related
            if intent in ['price_inquiry', 'product_info', 'availability']:
                product_matches = await self.search_products(
                    comment_text, comment_embedding, page_knowledge['products']
                )
                if product_matches:
                    relevant_info['products'] = product_matches
                    relevant_info['confidence'] = max(relevant_info['confidence'], 0.8)
            
            # Search FAQs
            if page_knowledge.get('faqs'):
                faq_matches = await self.search_faqs(
                    comment_text, comment_embedding, page_knowledge['faqs']
                )
                if faq_matches:
                    relevant_info['faqs'] = faq_matches
                    relevant_info['confidence'] = max(relevant_info['confidence'], 0.7)
            
            # Add business info based on intent
            if intent in ['hours', 'location', 'contact']:
                business_info = {}
                if intent == 'hours' and page_knowledge.get('business_hours'):
                    business_info['hours'] = page_knowledge['business_hours']
                elif intent == 'location' and page_knowledge.get('location'):
                    business_info['location'] = page_knowledge['location']
                elif intent == 'contact' and page_knowledge.get('contact_info'):
                    business_info['contact'] = page_knowledge['contact_info']
                
                if business_info:
                    relevant_info['business'] = business_info
                    relevant_info['confidence'] = max(relevant_info['confidence'], 0.9)
            
            return relevant_info
            
        except Exception as e:
            logger.error(f"Error finding relevant context: {e}")
            return {'confidence': 0.0}
    
    async def search_products(self, query: str, query_embedding, products: List[Dict]) -> List[Dict]:
        """Search for relevant products"""
        try:
            if not products:
                return []
            
            # Create product embeddings if not cached
            product_texts = []
            for product in products:
                text = f"{product['name']} {product['description']} {' '.join(product.get('keywords', []))}"
                product_texts.append(text)
            
            if not product_texts:
                return []
            
            product_embeddings = embedding_model.encode(product_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, product_embeddings)[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[-3:][::-1]  # Top 3 matches
            matches = []
            
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Similarity threshold
                    product = products[idx].copy()
                    product['similarity'] = float(similarities[idx])
                    matches.append(product)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []
    
    async def search_faqs(self, query: str, query_embedding, faqs: Dict) -> List[Dict]:
        """Search for relevant FAQs"""
        try:
            if not faqs:
                return []
            
            faq_questions = list(faqs.keys())
            if not faq_questions:
                return []
            
            faq_embeddings = embedding_model.encode(faq_questions)
            similarities = cosine_similarity(query_embedding, faq_embeddings)[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[-2:][::-1]  # Top 2 FAQs
            matches = []
            
            for idx in top_indices:
                if similarities[idx] > 0.4:  # FAQ similarity threshold
                    question = faq_questions[idx]
                    matches.append({
                        'question': question,
                        'answer': faqs[question],
                        'similarity': float(similarities[idx])
                    })
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching FAQs: {e}")
            return []

# Initialize context-aware AI
context_ai = ContextAwareAI()

async def load_page_knowledge_bases():
    """Load all page-specific knowledge bases"""
    try:
        pages = await db.page_configs.find().to_list(100)
        for page in pages:
            page_id = page['page_id']
            
            # Load page knowledge
            knowledge = await db.page_knowledge.find_one({'page_id': page_id})
            if knowledge:
                # Load products
                products = await db.products.find({'page_id': page_id}).to_list(1000)
                knowledge['products'] = products
                knowledge_bases[page_id] = knowledge
                logger.info(f"Loaded knowledge base for page: {page_id}")
        
        logger.info(f"Loaded {len(knowledge_bases)} page knowledge bases")
    except Exception as e:
        logger.error(f"Error loading knowledge bases: {e}")

async def generate_context_aware_reply(comment_data: dict, page_id: str, context_analysis: dict) -> str:
    """Generate context-aware reply based on comment analysis"""
    try:
        intent = context_analysis.get('intent', 'general')
        context = context_analysis.get('context', {})
        confidence = context_analysis.get('confidence', 0.0)
        
        # Build context-aware response
        response_parts = []
        
        # Handle product inquiries
        if intent == 'price_inquiry' and context.get('products'):
            product = context['products'][0]  # Top match
            if product.get('price'):
                response_parts.append(f"The {product['name']} is priced at ${product['price']:.2f}.")
            else:
                response_parts.append(f"Thanks for asking about {product['name']}! Please contact us for current pricing.")
        
        elif intent == 'product_info' and context.get('products'):
            product = context['products'][0]
            response_parts.append(f"Great question about {product['name']}! {product['description']}")
        
        elif intent == 'availability' and context.get('products'):
            product = context['products'][0]
            if product.get('availability', True):
                response_parts.append(f"Yes, {product['name']} is currently available!")
            else:
                response_parts.append(f"Sorry, {product['name']} is currently out of stock. We'll notify you when it's back!")
        
        # Handle business info
        elif intent == 'hours' and context.get('business', {}).get('hours'):
            response_parts.append(f"Our business hours are: {context['business']['hours']}")
        
        elif intent == 'location' and context.get('business', {}).get('location'):
            response_parts.append(f"You can find us at: {context['business']['location']}")
        
        elif intent == 'contact' and context.get('business', {}).get('contact'):
            response_parts.append(f"You can contact us at: {context['business']['contact']}")
        
        # Handle FAQs
        elif context.get('faqs'):
            faq = context['faqs'][0]  # Top FAQ match
            response_parts.append(faq['answer'])
        
        # Fallback to template-based response
        if not response_parts:
            page_config = await get_page_config(page_id)
            if page_config and page_config.response_templates.get(intent):
                templates = page_config.response_templates[intent]
                response_parts.append(templates[0])
            else:
                # Default responses by intent
                default_responses = {
                    'price_inquiry': "Thank you for your interest! Please contact us for detailed pricing information.",
                    'product_info': "Thanks for your question! We'd be happy to provide more details.",
                    'complaint': "We take all feedback seriously and want to make things right. Please contact us directly.",
                    'compliment': "Thank you so much for your kind words! We really appreciate your support.",
                    'general': "Thank you for your comment! We appreciate your engagement."
                }
                response_parts.append(default_responses.get(intent, default_responses['general']))
        
        # Combine response parts
        base_response = ' '.join(response_parts)
        
        # Paraphrase for naturalness
        paraphrases = await paraphrase_text(base_response, 1)
        final_response = paraphrases[0] if paraphrases else base_response
        
        return final_response
        
    except Exception as e:
        logger.error(f"Error generating context-aware reply: {e}")
        return "Thank you for your comment! We appreciate your engagement."

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
                    temperature=0.7 + (i * 0.1),
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

async def process_comment(comment_data: dict, page_id: str):
    """Enhanced comment processing with context awareness"""
    try:
        # Extract comment information
        comment_id = comment_data.get('id')
        message = comment_data.get('message', '')
        author_name = comment_data.get('from', {}).get('name', 'Unknown')
        author_id = comment_data.get('from', {}).get('id', '')
        post_id = comment_data.get('post_id', '')
        
        if not message:
            return
        
        # Get page config
        page_config = await get_page_config(page_id)
        if not page_config or not page_config.auto_reply_enabled:
            return
        
        # Context-aware analysis
        context_analysis = await context_ai.analyze_comment_context(message, page_id)
        
        # Only reply if confidence is above threshold
        if context_analysis['confidence'] < page_config.confidence_threshold:
            logger.info(f"Skipping reply - confidence {context_analysis['confidence']} below threshold {page_config.confidence_threshold}")
            return
        
        # Classify comment (legacy)
        classification, sentiment = await classify_comment(message)
        
        # Generate context-aware reply
        final_reply = await generate_context_aware_reply(comment_data, page_id, context_analysis)
        
        # Save comment data with enhanced context
        comment_obj = CommentData(
            post_id=post_id,
            comment_id=comment_id,
            page_id=page_id,
            comment_text=message,
            author_name=author_name,
            author_id=author_id,
            classification=classification,
            sentiment=sentiment,
            intent=context_analysis['intent'],
            context_match=context_analysis['context'],
            reply_text=final_reply,
            confidence_score=context_analysis['confidence']
        )
        
        await db.comments.insert_one(comment_obj.dict())
        
        # Post reply (demo mode for now)
        logger.info(f"DEMO MODE: Would reply to {author_name} with: {final_reply}")
        await db.comments.update_one(
            {"comment_id": comment_id},
            {"$set": {"replied": True}}
        )
        
        # Auto-learning: Store successful interaction
        if page_config.auto_learning_enabled:
            await store_learning_data(page_id, message, final_reply, context_analysis)
        
        logger.info(f"Context-aware reply generated for {author_name}: {context_analysis['intent']} -> {final_reply}")
        
    except Exception as e:
        logger.error(f"Error processing comment: {e}")

async def store_learning_data(page_id: str, comment: str, reply: str, context_analysis: dict):
    """Store successful interactions for learning"""
    try:
        learning_data = {
            'page_id': page_id,
            'comment': comment,
            'reply': reply,
            'intent': context_analysis['intent'],
            'context': context_analysis['context'],
            'confidence': context_analysis['confidence'],
            'timestamp': datetime.utcnow()
        }
        await db.learning_data.insert_one(learning_data)
    except Exception as e:
        logger.error(f"Error storing learning data: {e}")

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Context-Aware Facebook Auto-Reply System", "status": "running"}

# Knowledge Management
@api_router.post("/knowledge/{page_id}")
async def add_page_knowledge(page_id: str, knowledge: PageKnowledge):
    """Add or update page-specific knowledge"""
    try:
        knowledge.page_id = page_id
        await db.page_knowledge.update_one(
            {"page_id": page_id},
            {"$set": knowledge.dict()},
            upsert=True
        )
        
        # Reload knowledge base
        await load_page_knowledge_bases()
        
        return {"status": "success", "message": "Page knowledge updated"}
    except Exception as e:
        logger.error(f"Error adding page knowledge: {e}")
        raise HTTPException(status_code=500, detail="Error updating knowledge")

@api_router.post("/products/{page_id}")
async def add_products(page_id: str, products: List[ProductInfo]):
    """Add products for a specific page"""
    try:
        for product in products:
            product.page_id = page_id
            await db.products.insert_one(product.dict())
        
        # Reload knowledge base
        await load_page_knowledge_bases()
        
        return {"status": "success", "message": f"Added {len(products)} products"}
    except Exception as e:
        logger.error(f"Error adding products: {e}")
        raise HTTPException(status_code=500, detail="Error adding products")

@api_router.get("/knowledge/{page_id}")
async def get_page_knowledge(page_id: str):
    """Get page-specific knowledge"""
    try:
        knowledge = await db.page_knowledge.find_one({"page_id": page_id})
        products = await db.products.find({"page_id": page_id}).to_list(1000)
        
        if knowledge:
            knowledge['products'] = products
            return knowledge
        return {"message": "No knowledge found for this page"}
    except Exception as e:
        logger.error(f"Error getting page knowledge: {e}")
        raise HTTPException(status_code=500, detail="Error fetching knowledge")

# Enhanced demo endpoint
@api_router.post("/demo/comment")
async def demo_comment(request: dict):
    """Enhanced demo endpoint with context awareness"""
    try:
        comment_text = request.get('comment_text', '')
        page_id = request.get('page_id', 'demo_electronics_page')
        
        mock_comment = {
            'id': f'demo_comment_{uuid.uuid4()}',
            'message': comment_text,
            'from': {
                'name': 'Demo Customer',
                'id': 'demo_user_123'
            },
            'post_id': 'demo_post_123'
        }
        
        await process_comment(mock_comment, page_id)
        return {"status": "success", "message": "Context-aware demo comment processed"}
        
    except Exception as e:
        logger.error(f"Error in demo comment: {e}")
        raise HTTPException(status_code=500, detail="Error processing demo comment")

# Initialize demo data
@api_router.post("/demo/setup")
async def setup_demo_data():
    """Setup demo pages with realistic data"""
    try:
        # Electronics Store Demo Page
        electronics_config = PageConfig(
            page_id="demo_electronics_page",
            page_name="TechWorld Electronics",
            business_type="electronics",
            access_token="demo_token_electronics",
            confidence_threshold=0.6
        )
        
        electronics_knowledge = PageKnowledge(
            page_id="demo_electronics_page",
            page_name="TechWorld Electronics",
            business_type="electronics",
            business_hours="Mon-Fri: 9AM-8PM, Sat-Sun: 10AM-6PM",
            location="123 Tech Street, Silicon Valley, CA",
            contact_info="Phone: (555) 123-TECH, Email: info@techworld.com",
            faqs={
                "Do you offer warranty?": "Yes, we provide 1-year warranty on all electronics with free repair service.",
                "Do you deliver?": "Yes, we offer free delivery within 20 miles for orders above $100.",
                "What payment methods do you accept?": "We accept cash, credit cards, PayPal, and crypto payments."
            }
        )
        
        electronics_products = [
            ProductInfo(
                page_id="demo_electronics_page",
                name="iPhone 15 Pro",
                price=999.99,
                description="Latest iPhone with advanced camera system and A17 Pro chip",
                category="smartphones",
                keywords=["iphone", "apple", "smartphone", "phone", "mobile"]
            ),
            ProductInfo(
                page_id="demo_electronics_page",
                name="MacBook Air M3",
                price=1199.99,
                description="Ultra-thin laptop with M3 chip, perfect for professionals",
                category="laptops",
                keywords=["macbook", "apple", "laptop", "computer", "m3"]
            ),
            ProductInfo(
                page_id="demo_electronics_page",
                name="Sony WH-1000XM5",
                price=399.99,
                description="Premium noise-canceling wireless headphones",
                category="audio",
                keywords=["headphones", "sony", "wireless", "noise canceling", "audio"]
            )
        ]
        
        # Restaurant Demo Page
        restaurant_config = PageConfig(
            page_id="demo_restaurant_page",
            page_name="Bella Italia Restaurant",
            business_type="restaurant",
            access_token="demo_token_restaurant",
            confidence_threshold=0.6
        )
        
        restaurant_knowledge = PageKnowledge(
            page_id="demo_restaurant_page",
            page_name="Bella Italia Restaurant",
            business_type="restaurant",
            business_hours="Daily: 11AM-11PM",
            location="456 Food Avenue, Downtown, NY",
            contact_info="Phone: (555) 456-FOOD, Email: orders@bellaitalia.com",
            faqs={
                "Do you take reservations?": "Yes, we accept reservations for parties of 4 or more. Call us or book online.",
                "Do you have vegan options?": "Absolutely! We have a dedicated vegan menu with pasta, pizza, and dessert options.",
                "Do you deliver?": "Yes, we deliver within 5 miles. Free delivery on orders over $30."
            }
        )
        
        restaurant_products = [
            ProductInfo(
                page_id="demo_restaurant_page",
                name="Margherita Pizza",
                price=18.99,
                description="Classic pizza with fresh mozzarella, tomatoes, and basil",
                category="pizza",
                keywords=["pizza", "margherita", "cheese", "tomato", "basil"]
            ),
            ProductInfo(
                page_id="demo_restaurant_page",
                name="Spaghetti Carbonara",
                price=22.99,
                description="Traditional Roman pasta with eggs, pancetta, and parmesan",
                category="pasta",
                keywords=["pasta", "spaghetti", "carbonara", "eggs", "pancetta"]
            ),
            ProductInfo(
                page_id="demo_restaurant_page",
                name="Tiramisu",
                price=8.99,
                description="Classic Italian dessert with mascarpone and coffee",
                category="desserts",
                keywords=["tiramisu", "dessert", "mascarpone", "coffee", "italian"]
            )
        ]
        
        # Save all demo data
        await db.page_configs.update_one(
            {"page_id": "demo_electronics_page"},
            {"$set": electronics_config.dict()},
            upsert=True
        )
        
        await db.page_configs.update_one(
            {"page_id": "demo_restaurant_page"},
            {"$set": restaurant_config.dict()},
            upsert=True
        )
        
        await db.page_knowledge.update_one(
            {"page_id": "demo_electronics_page"},
            {"$set": electronics_knowledge.dict()},
            upsert=True
        )
        
        await db.page_knowledge.update_one(
            {"page_id": "demo_restaurant_page"},
            {"$set": restaurant_knowledge.dict()},
            upsert=True
        )
        
        # Clear existing products
        await db.products.delete_many({})
        
        # Add products
        for product in electronics_products + restaurant_products:
            await db.products.insert_one(product.dict())
        
        # Reload knowledge bases
        await load_page_knowledge_bases()
        
        return {
            "status": "success", 
            "message": "Demo data setup complete",
            "pages": ["demo_electronics_page", "demo_restaurant_page"],
            "products_added": len(electronics_products + restaurant_products)
        }
        
    except Exception as e:
        logger.error(f"Error setting up demo data: {e}")
        raise HTTPException(status_code=500, detail="Error setting up demo data")

# Existing routes (keeping them for compatibility)
@api_router.post("/paraphrase")
async def paraphrase_endpoint(request: dict):
    """Paraphrase text using AI"""
    try:
        text = request.get('text', '')
        num_paraphrases = request.get('num_paraphrases', 2)
        paraphrases = await paraphrase_text(text, num_paraphrases)
        return {"paraphrases": paraphrases}
    except Exception as e:
        logger.error(f"Error in paraphrase endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error generating paraphrases")

@api_router.get("/comments", response_model=List[CommentData])
async def get_comments(limit: int = 50):
    """Get recent comments with context"""
    try:
        comments = await db.comments.find().sort("timestamp", -1).limit(limit).to_list(limit)
        return [CommentData(**comment) for comment in comments]
    except Exception as e:
        logger.error(f"Error getting comments: {e}")
        raise HTTPException(status_code=500, detail="Error fetching comments")

@api_router.get("/pages", response_model=List[PageConfig])
async def get_pages():
    """Get all page configurations"""
    try:
        pages = await db.page_configs.find().to_list(1000)
        return [PageConfig(**page) for page in pages]
    except Exception as e:
        logger.error(f"Error getting pages: {e}")
        raise HTTPException(status_code=500, detail="Error fetching pages")

@api_router.post("/pages")
async def add_page(page_config: PageConfig):
    """Add or update page configuration"""
    try:
        await db.page_configs.update_one(
            {"page_id": page_config.page_id},
            {"$set": page_config.dict()},
            upsert=True
        )
        return {"status": "success", "message": "Page configuration saved"}
    except Exception as e:
        logger.error(f"Error saving page config: {e}")
        raise HTTPException(status_code=500, detail="Error saving page configuration")

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