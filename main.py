import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os

# Import ML model libraries
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic

# Import OpenAI for proxy endpoints
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False
    print("⚠️ OpenAI library not installed. Install with: pip install openai")

# Import python-dotenv to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
    print("✅ .env file loaded")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = None
if openai_available:
    try:
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        if OPENAI_API_KEY:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print("✅ OpenAI client initialized")
            print(f"   API Key: {OPENAI_API_KEY[:7]}...{OPENAI_API_KEY[-4:]}")  # Show partial key for verification
        else:
            print("⚠️ OPENAI_API_KEY not found in .env file")
            print("   Please create a .env file with: OPENAI_API_KEY=sk-...")
    except Exception as e:
        print(f"⚠️ Failed to initialize OpenAI: {e}")

# Request/Response models for worker search
class SearchRequest(BaseModel):
    description: str
    location: str

class WorkerResponse(BaseModel):
    worker_id: str
    worker_name: str
    service_type: str
    rating: float
    experience_years: int
    daily_wage_lkr: int
    phone_number: str
    email: str 
    city: str
    distance_km: float
    ai_confidence: float
    bio: str

class SearchResponse(BaseModel):
    workers: List[WorkerResponse]
    ai_analysis: Dict

# NEW: OpenAI Proxy Request/Response Models
class OpenAIImageRequest(BaseModel):
    base64_image: str
    mime_type: str
    user_message: Optional[str] = None

class OpenAITextRequest(BaseModel):
    message: str

class OpenAIResponse(BaseModel):
    content: str
    success: bool
    error: Optional[str] = None

# Location Name to Coordinates Converter
class LocationConverter:
    """Convert location names to coordinates"""
    
    def __init__(self):
        # Comprehensive Sri Lankan location database
        self.location_database = {
            # Major Cities
            'colombo': (6.9271, 79.8612),
            'kandy': (7.2906, 80.6337),
            'galle': (6.0535, 80.2210),
            'negombo': (7.2084, 79.8380),
            'jaffna': (9.6615, 80.0255),
            'kurunegala': (7.4818, 80.3609),
            'anuradhapura': (8.3114, 80.4037),
            'matara': (5.9549, 80.5550),
            'ratnapura': (6.6828, 80.3992),
            'trincomalee': (8.5874, 81.2152),
            'batticaloa': (7.7310, 81.6747),
            'badulla': (6.9934, 81.0550),
            'kalutara': (6.5854, 79.9607),
            'puttalam': (8.0362, 79.8283),
            'ampara': (7.2976, 81.6728),
            'kegalle': (7.2523, 80.3436),
            'polonnaruwa': (7.9403, 81.0188),
            'hambantota': (6.1429, 81.1212),
            'vavuniya': (8.7542, 80.4982),
            'mannar': (8.9810, 79.9044),
            'chilaw': (7.5759, 79.7955),
            'matale': (7.4675, 80.6234),
            'nuwara eliya': (6.9497, 80.7891),
            'bentota': (6.4258, 79.9951),
            'hikkaduwa': (6.1408, 80.1033),
            'tangalle': (6.0244, 80.7988),
            'mirissa': (5.9467, 80.4565),
            'arugam bay': (6.8414, 81.8361),
            'ella': (6.8667, 81.0467),
            'sigiriya': (7.9569, 80.7603),
            'dambulla': (7.8742, 80.6517),
            'mount lavinia': (6.8387, 79.8653),
            'dehiwala': (6.8518, 79.8729),
            'moratuwa': (6.7730, 79.8816),
            'sri jayawardenepura kotte': (6.9147, 79.9729),
            'maharagama': (6.8482, 79.9265),
            'wattala': (6.9889, 79.8917),
            'kelaniya': (6.9553, 79.9219),
            'katunayake': (7.1697, 79.8841)
        }
        
        print(f"✅ Location database loaded with {len(self.location_database)} locations")
    
    def get_coordinates(self, location_input):
        """Convert location name to coordinates"""
        normalized_name = location_input.lower().strip()
        
        # Direct match
        if normalized_name in self.location_database:
            return self.location_database[normalized_name]
        
        # Try partial matching
        for loc_key, coords in self.location_database.items():
            if normalized_name in loc_key or loc_key in normalized_name:
                return coords
        
        # Default to Colombo if not found
        print(f"⚠️ Location '{location_input}' not found, defaulting to Colombo")
        return (6.9271, 79.8612)
    
    def get_location_name(self, location_input):
        """Get properly formatted location name"""
        normalized_name = location_input.lower().strip()
        
        # Direct match
        if normalized_name in self.location_database:
            return normalized_name.title()
        
        # Try partial matching
        for loc_key in self.location_database.keys():
            if normalized_name in loc_key or loc_key in normalized_name:
                return loc_key.title()
        
        return "Colombo"  # Default

# AI Service Classifier
class AIServiceClassifier:
    """Advanced AI service classifier using semantic understanding"""

    def __init__(self):
        print("🧠 Initializing AI Service Classifier...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.label_encoder = LabelEncoder()
        self.trained = False

    def train(self, training_data):
        """Train using semantic embeddings and neural networks"""
        print("🎯 Training AI service classifier...")

        # Encode text to embeddings (semantic understanding)
        texts = training_data['problem_description'].tolist()
        labels = training_data['service_type'].tolist()

        print("📄 Generating semantic embeddings...")
        embeddings = self.sentence_model.encode(texts)

        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Train neural network on embeddings
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=1500,
            random_state=42
        )

        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, encoded_labels, test_size=0.2, random_state=42
        )

        print("📄 Training neural network...")
        self.classifier.fit(X_train, y_train)

        # Test accuracy
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✅ AI classifier trained!")
        print(f"🎯 Accuracy: {accuracy:.2%}")

        self.trained = True
        return accuracy

    def predict(self, text):
        """Predict service type using AI semantic understanding"""
        if not self.trained:
            raise Exception("AI model not trained yet!")

        # Generate semantic embedding
        text_embedding = self.sentence_model.encode([text])

        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(text_embedding)[0]
        predictions = self.classifier.predict(text_embedding)

        # Decode to service names
        service_names = self.label_encoder.inverse_transform(range(len(probabilities)))

        # Sort by confidence
        results = [(service, prob) for service, prob in zip(service_names, probabilities)]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:5]  # Top 5 predictions

# AI Location Extractor
class AILocationExtractor:
    """AI-based location extraction from text"""

    def __init__(self):
        print("📍 Initializing AI Location Extractor...")
        self.location_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.trained = False

    def train_location_model(self):
        """Train location extraction model"""
        print("🗺️ Training AI location extraction model...")

        # Sri Lankan locations with context
        self.location_data = {
            'colombo': (6.9271, 79.8612),
            'kandy': (7.2906, 80.6337),
            'galle': (6.0535, 80.2210),
            'negombo': (7.2084, 79.8380),
            'jaffna': (9.6615, 80.0255),
            'kurunegala': (7.4818, 80.3609),
            'anuradhapura': (8.3114, 80.4037),
            'matara': (5.9549, 80.5550),
            'ratnapura': (6.6828, 80.3992),
            'trincomalee': (8.5874, 81.2152),
            'batticaloa': (7.7310, 81.6747),
            'nuwara eliya': (6.9497, 80.7891),
            'mount lavinia': (6.8387, 79.8653),
            'moratuwa': (6.7730, 79.8816),
        }

        # Generate embeddings for location names
        self.location_names = list(self.location_data.keys())
        self.location_embeddings = self.location_model.encode(self.location_names)

        self.trained = True
        print(f"✅ Location model trained with {len(self.location_names)} locations")

    def extract_location(self, text):
        """Extract location using AI semantic similarity"""
        if not self.trained:
            self.train_location_model()

        # Generate semantic embedding for input text
        text_embedding = self.location_model.encode([text])

        # Calculate semantic similarities
        similarities = cosine_similarity(text_embedding, self.location_embeddings)[0]

        # Find best semantic match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]

        if best_similarity > 0.25:  # AI confidence threshold
            location_name = self.location_names[best_match_idx]
            coordinates = self.location_data[location_name]
            print(f"📍 AI detected location: {location_name.title()} (confidence: {best_similarity:.2f})")
            return coordinates, location_name
        else:
            print(f"📍 Location not clearly detected, using default (Colombo)")
            return (6.9271, 79.8612), "colombo"

# AI Time Extractor
class AITimeExtractor:
    """AI-based time requirement extraction"""

    def __init__(self):
        print("⏰ Initializing AI Time Extractor...")
        self.time_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.trained = False

    def train_time_model(self):
        """Train AI time extraction model"""
        print("🕑 Training AI time extraction model...")

        # Comprehensive time expressions
        time_training_examples = {
            'immediate': [
                'urgent help needed right now', 'emergency repair immediately',
                'asap need help', 'critical situation now', 'emergency service needed'
            ],
            'today': [
                'help needed today', 'service today please', 'can someone come today',
                'need help same day', 'today would be great'
            ],
            'tomorrow': [
                'tomorrow would work', 'help needed tomorrow', 'tomorrow morning preferred',
                'tomorrow afternoon fine', 'tomorrow evening works'
            ],
            'this_week': [
                'sometime this week', 'within the week', 'this week would work',
                'monday works', 'tuesday is good', 'wednesday preferred'
            ],
            'weekend': [
                'weekend would be better', 'saturday works', 'sunday is good',
                'weekend service needed', 'saturday morning preferred'
            ],
            'flexible': [
                'no rush', 'whenever convenient', 'flexible with timing',
                'anytime works', 'not urgent', 'no specific time'
            ]
        }

        # Create training data
        training_texts = []
        training_labels = []

        for time_type, examples in time_training_examples.items():
            for example in examples:
                training_texts.append(example)
                training_labels.append(time_type)

        # Generate embeddings
        embeddings = self.time_model.encode(training_texts)

        # Train classifier
        self.time_label_encoder = LabelEncoder()
        encoded_labels = self.time_label_encoder.fit_transform(training_labels)

        self.time_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.time_classifier.fit(embeddings, encoded_labels)

        self.trained = True
        print("✅ Time extraction model trained")

    def extract_time_requirement(self, text):
        """Extract time requirement using AI semantic understanding"""
        if not self.trained:
            self.train_time_model()

        # Generate semantic embedding
        text_embedding = self.time_model.encode([text])

        # AI prediction
        prediction = self.time_classifier.predict(text_embedding)[0]
        probabilities = self.time_classifier.predict_proba(text_embedding)[0]

        # Decode label
        time_type = self.time_label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)

        print(f"⏰ AI detected time: {time_type} (confidence: {confidence:.2f})")
        return time_type, confidence

# Complete AI Recommendation System
class CompleteAIRecommendationSystem:
    """Complete AI-powered recommendation system"""

    def __init__(self):
        print("🤖 Initializing Complete AI Recommendation System...")
        self.service_classifier = AIServiceClassifier()
        self.location_extractor = AILocationExtractor()
        self.time_extractor = AITimeExtractor()
        self.trained = False

    def train_complete_system(self, training_data):
        """Train all AI components"""
        print("🚀 Training Complete AI System...")

        # Train service classifier
        service_accuracy = self.service_classifier.train(training_data)

        # Train location and time extractors
        self.location_extractor.train_location_model()
        self.time_extractor.train_time_model()

        self.trained = True
        print("✅ Complete AI system trained successfully!")
        return service_accuracy

    def get_ai_recommendations(self, query, workers_data, user_lat, user_lng, max_results=5):
        """Get AI-powered worker recommendations"""
        if not self.trained:
            raise Exception("AI system not trained yet!")

        print(f"📍 AI PROCESSING: '{query}'")
        print("="*60)

        # AI analysis
        service_predictions = self.service_classifier.predict(query)
        location_coords, location_name = self.location_extractor.extract_location(query)
        time_type, time_confidence = self.time_extractor.extract_time_requirement(query)

        print(f"🎯 AI Service Analysis: {[(s, f'{p:.1%}') for s, p in service_predictions]}")
        print(f"📍 AI Location: {location_name.title()} {location_coords}")
        print(f"⏰ AI Time: {time_type} (confidence: {time_confidence:.2f})")
        print()

        # Use user's actual coordinates
        user_coords = (user_lat, user_lng)

        # Convert to DataFrame if needed
        if isinstance(workers_data, dict):
            workers_df = pd.DataFrame(workers_data['workers'])
        else:
            workers_df = workers_data

        # Filter workers using AI predictions
        likely_services = [s for s, p in service_predictions if p > 0.1]
        relevant_workers = workers_df[workers_df['service_type'].isin(likely_services)].copy()

        if len(relevant_workers) == 0:
            likely_services = [service_predictions[0][0]]
            relevant_workers = workers_df[workers_df['service_type'].isin(likely_services)].copy()

        if len(relevant_workers) == 0:
            print("⚠ No suitable workers found")
            return [], {"error": "No suitable workers found"}

        # AI-powered scoring
        scored_workers = []
        for idx, worker in relevant_workers.iterrows():
            # Service match score (AI confidence)
            service_confidence = next((p for s, p in service_predictions if s == worker['service_type']), 0)
            service_score = service_confidence * 70  # 70 points max

            # Distance score (geographic proximity)
            worker_coords = (worker['location']['latitude'], worker['location']['longitude'])
            distance = geodesic(user_coords, worker_coords).kilometers
            distance_score = max(0, 20 - (distance * 0.2))  # 20 points max

            # Quality score (rating & experience)
            quality_score = (worker['rating'] / 5.0) * 10  # 10 points max

            total_score = service_score + distance_score + quality_score

            scored_workers.append({
                'worker': worker,
                'score': total_score,
                'distance_km': distance,
                'service_confidence': service_confidence
            })

        # Sort by AI score
        scored_workers.sort(key=lambda x: x['score'], reverse=True)

        return scored_workers[:max_results], {
            'service_predictions': [(s, f'{p:.1%}') for s, p in service_predictions],
            'detected_location': location_name,
            'time_requirement': time_type,
            'user_location': user_coords
        }

def create_ai_training_data():
    """Create comprehensive training data for AI learning"""
    print("🎯 Creating AI training data with semantic understanding...")

    # Real-world problem descriptions with context
    service_training_examples = {
        'plumbing': [
            'water is dripping from my bathroom faucet constantly',
            'toilet keeps overflowing and making a huge mess in bathroom',
            'kitchen sink drain is completely blocked and water backing up',
            'shower has no water pressure at all very frustrating',
            'water heater stopped working no hot water for days',
            'pipe burst in basement flooding the entire area',
            'bathroom sink leaking underneath causing water damage',
            'toilet not flushing properly waste backing up',
            'water pressure extremely low throughout entire house',
            'drain making gurgling sounds and smells terrible',
            'faucet handle broken cannot turn water on or off',
            'water bill very high suspect underground leak'
        ],
        'electrical': [
            'lights keep flickering throughout the entire house',
            'power outlet stopped working completely no electricity',
            'circuit breaker keeps tripping shutting off power repeatedly',
            'electrical switch making loud sparking sounds very dangerous',
            'half the house has no electricity sudden power loss',
            'ceiling fan stopped working motor making grinding noise',
            'electrical wiring looks old and potentially hazardous',
            'light fixtures need professional installation help',
            'power surges damaging electronic devices frequently',
            'electrical panel needs upgrade current system outdated',
            'smoke detector not working needs replacement urgently',
            'outdoor lighting system completely non functional'
        ],
        'carpentry': [
            'wooden door frame rotted and falling apart badly',
            'cabinet doors hanging loose and not closing properly',
            'wooden fence panels broken need replacement urgently',
            'deck boards warped and becoming dangerous to walk on',
            'stairs creaking loudly and feel unstable unsafe',
            'wooden furniture repair needed broken leg on table',
            'custom shelving installation needed for storage',
            'window frames damaged by water need repair',
            'wooden floor squeaking badly needs fixing',
            'crown molding installation needed for living room',
            'door not closing properly frame issues',
            'built-in cabinet construction needed bedroom'
        ],
        'painting': [
            'entire house needs repainting walls look terrible',
            'exterior paint peeling off badly looks awful',
            'ceiling water stains need covering and painting',
            'bedroom walls have cracks need repair and paint',
            'kitchen cabinets need repainting old and faded',
            'living room accent wall painting needed',
            'bathroom walls moldy need treatment and repainting',
            'wooden fence needs staining and weatherproofing',
            'garage floor needs epoxy coating application',
            'deck needs restaining protection from weather',
            'interior door painting needed multiple rooms',
            'textured wall finish application needed professional'
        ],
        'hvac': [
            'air conditioner not cooling house is too hot',
            'heating system not working freezing in winter',
            'ac making loud rattling noise constantly',
            'hvac unit leaking water all over floor',
            'thermostat not working properly temperature issues',
            'air filter needs changing poor air quality',
            'ventilation system cleaning needed dust everywhere',
            'heat pump not functioning efficiently high bills',
            'ac compressor failed need immediate replacement',
            'ductwork leaking air losing efficiency',
            'furnace making strange noises concerning',
            'central heating not reaching all rooms'
        ],
        'masonry': [
            'brick wall cracking badly structural concern',
            'chimney damaged needs professional repair work',
            'concrete driveway cracked and uneven dangerous',
            'stone patio needs repair loose stones everywhere',
            'retaining wall collapsing needs urgent attention',
            'brick fireplace repair needed mortar crumbling',
            'concrete walkway settled unevenly tripping hazard',
            'stone veneer installation needed exterior upgrade',
            'brick pointing needed mortar deteriorating',
            'concrete foundation cracks need professional assessment',
            'patio pavers sinking and shifting badly',
            'stone steps damaged need reconstruction work'
        ],
        'roofing': [
            'roof leaking badly water coming inside house',
            'shingles missing after storm need replacement',
            'roof sagging looks dangerous structural issue',
            'gutter system clogged water overflowing badly',
            'skylight leaking water damage on ceiling',
            'flat roof pooling water drainage problems',
            'roof tiles broken need urgent repair',
            'attic insulation inadequate heat loss problem',
            'roof ventilation poor causing moisture issues',
            'chimney flashing damaged causing leaks',
            'metal roof panels loose rattling in wind',
            'roof inspection needed before selling house'
        ],
        'flooring': [
            'floor tiles cracked and broken need replacement',
            'hardwood floor refinishing needed badly scratched',
            'laminate flooring buckling water damage suspected',
            'tile grout dirty and discolored needs cleaning',
            'carpet worn out needs complete replacement',
            'vinyl flooring peeling at edges looks terrible',
            'floor leveling needed before new installation',
            'hardwood floor installation wanted bedroom',
            'tile floor installation needed bathroom',
            'marble floor polishing needed dull and scratched',
            'epoxy floor coating wanted garage',
            'subfloor repair needed squeaking sounds'
        ]
    }

    # Generate training dataset
    training_data = []
    for service_type, examples in service_training_examples.items():
        for example in examples:
            training_data.append({
                'problem_description': example,
                'service_type': service_type
            })

    training_df = pd.DataFrame(training_data)
    print(f"✅ Generated {len(training_df)} training examples")
    return training_df

# Global variables
ai_system = None
workers_database = None
location_converter = None

@app.on_event("startup")
async def startup_event():
    global ai_system, workers_database, location_converter
    
    # Initialize Location Converter
    location_converter = LocationConverter()
    print("✅ Location Converter initialized")
    
    # Initialize AI system
    ai_system = CompleteAIRecommendationSystem()
    training_data = create_ai_training_data()
    accuracy = ai_system.train_complete_system(training_data)
    
    # Load workers database
    try:
        with open('handyman_database_3000.json', 'r', encoding='utf-8') as f:
            workers_database = json.load(f)
        print(f"✅ Loaded {len(workers_database['workers'])} workers from database")
    except FileNotFoundError:
        print("⚠️ Database file not found!")
        workers_database = {'workers': []}

# NEW: OpenAI Proxy Endpoints
@app.post("/api/openai/analyze-image", response_model=OpenAIResponse)
async def proxy_analyze_image(request: OpenAIImageRequest):
    """Proxy endpoint for OpenAI image analysis - fixes CORS issues"""
    if not openai_client:
        return OpenAIResponse(
            content="",
            success=False,
            error="OpenAI client not initialized. Please set OPENAI_API_KEY environment variable."
        )
    
    try:
        # Prepare the request
        user_message = request.user_message or "Please analyze this image and describe what you see. If this appears to be a maintenance or repair issue, provide details about what might be wrong and potential solutions."
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_message
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{request.mime_type};base64,{request.base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        content = response.choices[0].message.content
        
        return OpenAIResponse(
            content=content,
            success=True
        )
        
    except Exception as e:
        return OpenAIResponse(
            content="",
            success=False,
            error=f"Failed to analyze image: {str(e)}"
        )

@app.post("/api/openai/send-message", response_model=OpenAIResponse)
async def proxy_send_message(request: OpenAITextRequest):
    """Proxy endpoint for OpenAI text chat - fixes CORS issues"""
    if not openai_client:
        return OpenAIResponse(
            content="",
            success=False,
            error="OpenAI client not initialized. Please set OPENAI_API_KEY environment variable."
        )
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for a home services marketplace app called FixMate. Help users understand their maintenance and repair issues, and provide guidance on what type of service they might need."
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        
        return OpenAIResponse(
            content=content,
            success=True
        )
        
    except Exception as e:
        return OpenAIResponse(
            content="",
            success=False,
            error=f"Failed to send message: {str(e)}"
        )

@app.get("/")
async def root():
    return {
        "message": "Handyman AI Service API", 
        "status": "running", 
        "workers_count": len(workers_database['workers']) if workers_database else 0,
        "location_converter": "enabled",
        "email_support": "enabled",
        "openai_proxy": "enabled" if openai_client else "disabled",
        "endpoints": {
            "worker_search": "/search",
            "image_analysis": "/api/openai/analyze-image",
            "text_chat": "/api/openai/send-message"
        }
    }
    
@app.post("/search", response_model=SearchResponse)
async def search_workers(request: SearchRequest):
    if not ai_system or not workers_database or not location_converter:
        raise HTTPException(status_code=500, detail="AI system not initialized")
    
    try:
        user_lat, user_lng = location_converter.get_coordinates(request.location)
        location_name = location_converter.get_location_name(request.location)
        
        print(f"🌍 User location: {request.location} -> ({user_lat}, {user_lng})")
        
        recommendations, analysis = ai_system.get_ai_recommendations(
            request.description, 
            workers_database,
            user_lat, 
            user_lng
        )
        
        analysis['user_input_location'] = location_name
        
        # Extract email from worker data
        worker_responses = []
        for rec in recommendations:
            worker = rec['worker']
            
            # Extract email from dataset
            worker_email = worker.get('contact', {}).get('email', '')
            if not worker_email:
                worker_email = worker.get('email', '')
            if not worker_email:
                # Fallback only if email not in dataset
                worker_email = f"{worker['worker_id'].lower()}@fixmate.worker"
            
            worker_responses.append(WorkerResponse(
                worker_id=worker['worker_id'],
                worker_name=worker['worker_name'],
                service_type=worker['service_type'],
                rating=worker['rating'],
                experience_years=worker['experience_years'],
                daily_wage_lkr=worker['pricing']['daily_wage_lkr'],
                phone_number=worker['contact']['phone_number'],
                email=worker_email,
                city=worker['location']['city'],
                distance_km=round(rec['distance_km'], 1),
                ai_confidence=round(rec['service_confidence'] * 100, 1),
                bio=worker['profile']['bio']
            ))
        
        return SearchResponse(workers=worker_responses, ai_analysis=analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)