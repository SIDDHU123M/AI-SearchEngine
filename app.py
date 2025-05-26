from flask import Flask, render_template, request, jsonify, send_from_directory, abort
import os
import time
import logging
import hashlib
import json
import traceback
import speech_recognition as sr
from threading import Lock
from datetime import datetime, timedelta
from functools import lru_cache
from dotenv import load_dotenv
from main_engine import main as run_engine

# ===== APPLICATION CONFIGURATION =====
# Configure logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AI-SearchEngine")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, 
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')

# Determine environment and configure accordingly
DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')

# Cache configuration
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Bookmarks configuration
BOOKMARKS_DIR = "bookmarks"
os.makedirs(BOOKMARKS_DIR, exist_ok=True)

# Cache lock for thread safety
cache_lock = Lock()
cache_stats = {"hits": 0, "entries": 0}

# Initialize speech recognition
recognizer = sr.Recognizer()

# ===== CONTENT SAFETY FUNCTIONS =====
def contains_prohibited_content(text):
    """Check if text contains prohibited content"""
    # Basic implementation - could be expanded with more sophisticated checks
    prohibited_terms = [
        "porn", "illegal drugs", "suicide", "self-harm", "terrorism", 
        "hate speech", "violence", "child abuse", "weapons"
    ]
    
    text_lower = text.lower()
    for term in prohibited_terms:
        if term in text_lower:
            return True
    return False

# ===== CACHE MANAGEMENT FUNCTIONS =====
def get_cache_key(query):
    """Generate a cache key for a query"""
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def get_cached_result(query):
    """Retrieve a cached result for the query if it exists"""
    global cache_stats
    cache_key = get_cache_key(query)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    if os.path.exists(cache_file):
        try:
            with cache_lock:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Check if cache is still valid (24 hours)
                timestamp = cache_data.get('timestamp', 0)
                if time.time() - timestamp < 86400:  # 24 hours
                    cache_stats["hits"] += 1
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    return cache_data.get('result')
                else:
                    # Cache expired
                    os.remove(cache_file)
                    return None
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")
            return None
    return None

def save_to_cache(query, result):
    """Save a result to the cache"""
    global cache_stats
    cache_key = get_cache_key(query)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    try:
        with cache_lock:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': query,
                    'result': result,
                    'timestamp': time.time()
                }, f, ensure_ascii=False)
            
            # Update stats
            cache_stats["entries"] = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
            logger.info(f"Saved result to cache for query: {query[:50]}...")
    except Exception as e:
        logger.error(f"Error saving to cache: {str(e)}")

def count_sources(result_text):
    """Count the number of sources in a result"""
    if not result_text or "Sources:" not in result_text:
        return 0
    
    sources_section = result_text.split("Sources:")[-1]
    return sources_section.count("- ")

# ===== BOOKMARK FUNCTIONS =====
def save_bookmark(query, result):
    """Save a bookmark"""
    # First, check for existing bookmarks with the same query
    existing_bookmark = None
    for filename in os.listdir(BOOKMARKS_DIR):
        if filename.endswith('.json'):
            file_path = os.path.join(BOOKMARKS_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    bookmark_data = json.load(f)
                    if bookmark_data.get('query') == query:
                        existing_bookmark = file_path
                        break
            except Exception as e:
                logger.error(f"Error reading bookmark file {filename}: {str(e)}")
                continue
    
    # If existing bookmark found, update it instead of creating a new one
    if existing_bookmark:
        try:
            with open(existing_bookmark, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': query,
                    'result': result,
                    'timestamp': os.path.basename(existing_bookmark).replace('.json', '')
                }, f, ensure_ascii=False)
            logger.info(f"Updated bookmark for query: {query[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error updating bookmark: {str(e)}")
            return False
    
    # Otherwise create a new bookmark
    bookmark_file = os.path.join(BOOKMARKS_DIR, f"{int(time.time())}.json")
    try:
        with open(bookmark_file, 'w', encoding='utf-8') as f:
            json.dump({
                'query': query,
                'result': result,
                'timestamp': int(time.time())
            }, f, ensure_ascii=False)
        logger.info(f"Saved bookmark for query: {query[:50]}...")
        return True
    except Exception as e:
        logger.error(f"Error saving bookmark: {str(e)}")
        return False

def get_bookmarks():
    """Get all bookmarks"""
    bookmarks = []
    
    try:
        for filename in os.listdir(BOOKMARKS_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(BOOKMARKS_DIR, filename), 'r', encoding='utf-8') as f:
                    bookmark = json.load(f)
                    bookmarks.append(bookmark)
        
        # Sort by timestamp descending
        bookmarks.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return bookmarks
    except Exception as e:
        logger.error(f"Error getting bookmarks: {str(e)}")
        return []

# ===== VOICE RECOGNITION FUNCTIONS =====
def recognize_speech():
    """Recognize speech from microphone"""
    try:
        with sr.Microphone() as source:
            logger.info("Listening for speech...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
        logger.info("Speech captured, recognizing...")
        text = recognizer.recognize_google(audio)
        logger.info(f"Recognized speech: {text}")
        return text
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {str(e)}")
        return None
    except sr.UnknownValueError:
        logger.error("Speech recognition could not understand audio")
        return None
    except Exception as e:
        logger.error(f"Speech recognition error: {str(e)}")
        return None

# ===== FLASK ROUTES =====
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Process a search query"""
    start_time = time.time()
    request_data = request.get_json()
    
    if not request_data:
        return jsonify({'error': 'Invalid request format'}), 400
        
    user_query = request_data.get('query', '').strip()
    
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    # Log the incoming query
    logger.info(f"Processing query: {user_query[:50]}{'...' if len(user_query) > 50 else ''}")
    
    # Check for unsafe or prohibited queries
    if contains_prohibited_content(user_query):
        logger.warning(f"Blocked prohibited query: {user_query}")
        return jsonify({
            'error': 'This query contains prohibited content and cannot be processed.'
        }), 403
    
    # Check cache for existing results
    cached_result = get_cached_result(user_query)
    if cached_result:
        elapsed_time = time.time() - start_time
        return jsonify({
            'result': cached_result, 
            'time': f"{elapsed_time:.2f}", 
            'cached': True,
            'stats': {
                'cache_hits': cache_stats["hits"],
                'cache_size': cache_stats["entries"]
            }
        })
    
    # Process the query
    try:
        # Execute search and analysis
        result = run_engine(user_query)
        elapsed_time = time.time() - start_time
        
        # Save result to cache
        save_to_cache(user_query, result)
        
        # Send detailed response to client
        return jsonify({
            'result': result, 
            'time': f"{elapsed_time:.2f}",
            'cached': False,
            'stats': {
                'processing_time': f"{elapsed_time:.2f}s",
                'sources_count': count_sources(result)
            }
        })
    except Exception as e:
        # Log the full error with traceback
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a user-friendly error message
        return jsonify({
            'error': 'An error occurred while processing your query. Please try again or rephrase your question.',
            'details': str(e) if DEBUG_MODE else None
        }), 500

@app.route('/voice', methods=['POST'])
def voice_search():
    """Process voice input for search"""
    try:
        text = recognize_speech()
        if not text:
            return jsonify({
                'error': 'Could not recognize speech. Please try again.'
            }), 400
            
        return jsonify({
            'recognized_text': text
        })
    except Exception as e:
        logger.error(f"Error processing voice: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': 'An error occurred while processing your voice input. Please try again.',
            'details': str(e) if DEBUG_MODE else None
        }), 500

@app.route('/bookmark', methods=['POST'])
def add_bookmark():
    """Add a bookmark"""
    request_data = request.get_json()
    
    if not request_data:
        return jsonify({'error': 'Invalid request format'}), 400
        
    query = request_data.get('query', '').strip()
    result = request_data.get('result', '').strip()
    
    if not query or not result:
        return jsonify({'error': 'Query and result are required'}), 400
    
    success = save_bookmark(query, result)
    
    if success:
        return jsonify({'message': 'Bookmark saved successfully'})
    else:
        return jsonify({'error': 'Failed to save bookmark'}), 500

@app.route('/bookmarks', methods=['GET'])
def get_all_bookmarks():
    """Get all bookmarks"""
    bookmarks = get_bookmarks()
    return jsonify({'bookmarks': bookmarks})

@app.route('/bookmark/delete', methods=['POST'])
def delete_bookmark():
    """Delete a bookmark"""
    request_data = request.get_json()
    
    if not request_data:
        return jsonify({'error': 'Invalid request format'}), 400
        
    timestamp = request_data.get('timestamp')
    
    if not timestamp:
        return jsonify({'error': 'Timestamp is required'}), 400
    
    try:
        bookmark_file = os.path.join(BOOKMARKS_DIR, f"{timestamp}.json")
        if os.path.exists(bookmark_file):
            os.remove(bookmark_file)
            logger.info(f"Deleted bookmark with timestamp: {timestamp}")
            return jsonify({'message': 'Bookmark deleted successfully'})
        else:
            return jsonify({'error': 'Bookmark not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting bookmark: {str(e)}")
        return jsonify({'error': 'Failed to delete bookmark'}), 500

# ===== APPLICATION ENTRY POINT =====
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=DEBUG_MODE)