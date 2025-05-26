import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from readability import Document
import json
import time
import re
import os
import logging
import concurrent.futures
import multiprocessing
import httpx
import html2text
import urllib.parse
from datetime import datetime
from functools import lru_cache
from openai import OpenAI
from dotenv import load_dotenv

# ===== CONFIGURATION AND SETUP =====
# Set up logging
logger = logging.getLogger("AI-SearchEngine.Engine")

# Load environment variables for API keys
load_dotenv()
api_key = os.getenv('openaiAPI')
if not api_key:
    logger.error("OpenAI API key not found. Set the 'openaiAPI' environment variable.")
    raise ValueError("OpenAI API key not found")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# ===== REGEX PATTERNS FOR TEXT PROCESSING =====
# Compile patterns once for better performance
WHITESPACE_PATTERN = re.compile(r'\s+')
AD_PATTERN = re.compile(r'accept cookies|privacy policy|terms of use|subscribe to our newsletter|sign up|log in|related articles', re.IGNORECASE)
URL_PATTERN = re.compile(r'https?://\S+')
DATE_PATTERN = re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', re.IGNORECASE)
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
BULLET_PATTERN = re.compile(r'•|\*|–|-|→')

# ===== KNOWN HIGH-QUALITY DOMAINS =====
HIGH_QUALITY_DOMAINS = {
    # News & Analysis
    'nytimes.com': 0.9, 'wsj.com': 0.9, 'reuters.com': 0.95, 'bloomberg.com': 0.9,
    'apnews.com': 0.95, 'theguardian.com': 0.9, 'bbc.com': 0.9, 'economist.com': 0.9,
    'ft.com': 0.9, 'npr.org': 0.9, 'washingtonpost.com': 0.9,
    
    # Science & Academic
    'nature.com': 0.95, 'science.org': 0.95, 'springer.com': 0.9, 'ncbi.nlm.nih.gov': 0.95,
    'nejm.org': 0.95, 'sciencedirect.com': 0.9, 'pnas.org': 0.95, 'ieee.org': 0.9,
    'plos.org': 0.9, 'cell.com': 0.95, 'oup.com': 0.9, 'wiley.com': 0.9,
    
    # Tech & Industry
    'techcrunch.com': 0.85, 'arstechnica.com': 0.85, 'wired.com': 0.85, 'hbr.org': 0.9,
    'zdnet.com': 0.8, 'cnet.com': 0.8, 'technologyreview.com': 0.9, 'forbes.com': 0.8,
    
    # Reference
    'wikipedia.org': 0.85, 'britannica.com': 0.9, 'merriam-webster.com': 0.9, 
    'mayoclinic.org': 0.95, 'cdc.gov': 0.95, 'nih.gov': 0.95, 'who.int': 0.95,
    'nasa.gov': 0.95, 'un.org': 0.9, 'worldbank.org': 0.9, 'imf.org': 0.9
}

# ===== TEXT PROCESSING FUNCTIONS =====
def clean_text(text):
    """Clean text by removing ads, excess whitespace, URLs, and email addresses"""
    if not text:
        return ""
        
    # Remove excessive whitespace
    text = WHITESPACE_PATTERN.sub(' ', text)
    
    # Remove common advertisement phrases
    text = AD_PATTERN.sub('', text)
    
    # Remove URLs and email addresses
    text = URL_PATTERN.sub('', text)
    text = EMAIL_PATTERN.sub('', text)
    
    # Remove very short lines (likely navigation elements)
    lines = text.split('\n')
    lines = [line for line in lines if len(line.strip()) > 30 or BULLET_PATTERN.search(line)]
    text = '\n'.join(lines)
    
    # Remove duplicate lines
    unique_lines = []
    seen_lines = set()
    for line in text.split('\n'):
        clean_line = line.strip()
        if clean_line and clean_line not in seen_lines:
            unique_lines.append(line)
            seen_lines.add(clean_line)
    
    text = '\n'.join(unique_lines)
    return text.strip()

# ===== SEARCH AND WEB SCRAPING FUNCTIONS =====
@lru_cache(maxsize=128)
def search_links(query, max_results=12):
    """Search for relevant links using DuckDuckGo"""
    logger.info(f"Searching for links related to: {query}")
    start_time = time.time()
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results+4))
        
        # Filter out low-quality or irrelevant results
        filtered_results = []
        seen_domains = set()
        
        # Prioritize high-quality domains and ensure diversity
        for result in results:
            url = result.get('href', '')
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc.lower()
            domain = domain.replace('www.', '')
            
            # Skip if from a domain we already have (to ensure diversity)
            if domain in seen_domains and len(filtered_results) >= 3:
                continue
                
            # Add to filtered results
            filtered_results.append({
                'title': result.get('title', ''),
                'url': url,
                'domain': domain
            })
            seen_domains.add(domain)
            
            # Break if we have enough results
            if len(filtered_results) >= max_results:
                break
        
        logger.info(f"Found {len(filtered_results)} links in {time.time() - start_time:.2f}s")
        return filtered_results
    except Exception as e:
        logger.error(f"Error searching for links: {str(e)}")
        return []

def fetch_text_from_url(url):
    """Fetch and extract clean text from a URL with optimized error handling"""
    start_time = time.time()
    logger.debug(f"Fetching content from: {url}")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Make the request with timeout
        res = requests.get(url, timeout=10, headers=headers)
        res.raise_for_status()
        
        # Check if the content is HTML
        content_type = res.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            logger.debug(f"Skipping non-HTML content from {url}")
            return None
        
        # Get domain quality score
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc.lower()
        domain = domain.replace('www.', '')
        quality_score = HIGH_QUALITY_DOMAINS.get(domain, 0.7)  # Default score for unknown domains
        
        # Extract publish date if available
        publish_date = None
        date_match = DATE_PATTERN.search(res.text)
        if date_match:
            publish_date = date_match.group(0)
        
        # Use readability to extract main content
        doc = Document(res.text)
        html = doc.summary()
        title = doc.title()
        
        # Use lxml parser for faster parsing
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove unwanted tags
        for tag in soup(['header', 'footer', 'nav', 'style', 'script', 'noscript', 'aside', 'iframe', 'form']):
            tag.decompose()
        
        # Get text from readability extraction
        text = soup.get_text(separator="\n", strip=True)
        text = clean_text(text)
        
        # If content is too short, try a second approach with html2text
        if len(text) < 1000:
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.ignore_tables = False
            h.body_width = 0  # No wrapping
            text2 = h.handle(res.text)
            text2 = clean_text(text2)
            
            # Choose the better content
            if len(text2) > len(text) * 1.5:
                text = text2
        
        # Determine if it's likely a news/article site
        is_news = any(news_term in domain.lower() for news_term in 
                   ['news', 'times', 'post', 'tribune', 'herald', 'guardian', 
                    'journal', 'daily', 'cnn', 'bbc', 'nbc', 'abc', 'cbs', 'fox'])
        
        # Create result object with metadata
        result = {
            "url": url, 
            "title": title, 
            "text": text,
            "site_name": domain,
            "publish_date": publish_date,
            "is_news": is_news,
            "content_length": len(text),
            "quality_score": quality_score,
            "fetch_time": time.time() - start_time
        }
        
        # Only return if there's meaningful content
        if len(text) > 300:  # Minimum content length
            logger.debug(f"Successfully extracted {len(text)} chars from {url} in {result['fetch_time']:.2f}s")
            return result
        else:
            logger.debug(f"Content too short from {url}: {len(text)} chars")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.debug(f"Request error for {url}: {str(e)}")
        return None
    except Exception as e:
        logger.debug(f"Error processing {url}: {str(e)}")
        return None

# ===== CONTENT ANALYSIS FUNCTIONS =====
def extract_keywords(text, max_words=5):
    """Extract likely keywords from text"""
    # Basic implementation - could be improved with NLP
    words = re.findall(r'\b[a-zA-Z]{4,15}\b', text.lower())
    word_counts = {}
    
    for word in words:
        if word not in ['that', 'this', 'with', 'from', 'have', 'more', 'what', 'when', 'where', 'will', 'also']:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_words]]

def rank_articles(articles, query):
    """Rank articles by relevance to query, quality, and other factors"""
    query_keywords = set(extract_keywords(query))
    
    for article in articles:
        # Calculate relevance score
        article_keywords = set(extract_keywords(article['text'], max_words=10))
        keyword_overlap = len(query_keywords.intersection(article_keywords))
        
        # Calculate recency bonus (if available)
        recency_bonus = 0
        if article.get('publish_date'):
            try:
                date_str = article['publish_date']
                date_obj = datetime.strptime(date_str, '%b %d, %Y')
                days_ago = (datetime.now() - date_obj).days
                if days_ago < 30:  # Published within the last month
                    recency_bonus = 0.2 * (30 - days_ago) / 30
            except:
                pass
        
        # Calculate final score
        article['relevance_score'] = (
            (0.5 * keyword_overlap) +                  # Keyword relevance
            (0.3 * article.get('quality_score', 0.7)) + # Domain quality
            (0.1 * min(article.get('content_length', 0) / 5000, 1)) + # Content length (capped)
            recency_bonus                              # Recency bonus
        )
    
    # Sort by relevance score
    return sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True)

# ===== OPENAI API INTEGRATION =====
def process_with_chatgpt(user_query, articles, max_tokens=1500):
    """Process search results with OpenAI's API to generate a response with related queries"""
    if not articles:
        return "I couldn't find relevant information. Please try a different search query."
        
    # Determine if this is a current events or factual query
    is_current_event = any(word in user_query.lower() for word in 
                        ["latest", "recent", "new", "current", "today", "yesterday", 
                         "week", "month", "update", "news", "announced", "released"])
    
    # Prepare combined text from ranked articles
    combined_text = ""
    sources = []
    
    # Process articles with diversity and quality in mind
    selected_articles = []
    domains_selected = set()
    
    # First pass: select high-quality articles while maintaining domain diversity
    for article in articles:
        domain = article.get('site_name', '')
        if len(domains_selected) < 3 or domain not in domains_selected:
            selected_articles.append(article)
            domains_selected.add(domain)
            if len(selected_articles) >= 7:
                break
    
    # Add articles to context
    for i, article in enumerate(selected_articles):
        # Determine recency string if publish date is available
        recency_info = ""
        if article.get('publish_date'):
            recency_info = f"Published: {article.get('publish_date')}"
        
        # Adjust text length based on article relevance and position
        # More relevant articles get more space in the context
        if i < 3:  # Top 3 articles get more space
            preview_length = min(2500, len(article['text']))
        else:
            preview_length = min(1500, len(article['text']))
            
        preview_text = article['text'][:preview_length]
        
        # Add to combined text with better formatting
        combined_text += f"SOURCE {i+1}: {article['site_name']} - {article['title']}\n"
        if recency_info:
            combined_text += f"{recency_info}\n"
        combined_text += f"URL: {article['url']}\n\n{preview_text}\n\n---\n\n"
        
        # Add to sources list
        source_entry = f"[{article['site_name']}] - {article['title']} ({article['url']})"
        sources.append(source_entry)

    # Craft an appropriate system prompt based on query type
    if is_current_event:
        system_prompt = f"""
You are an advanced AI research assistant that provides comprehensive and up-to-date information.

The user has asked: "{user_query}"

This appears to be a query about recent events or developments. Follow these principles:
1. Respond with the MOST RECENT information from the provided sources
2. Emphasize dates and timeframes to establish when events occurred
3. Note any conflicting information between sources and which is more recent
4. Organize information chronologically when appropriate
5. Indicate if the information might be outdated
6. Directly quote important statements or statistics from sources when relevant
7. Do NOT include information that isn't found in the provided sources
8. If sources are insufficient to fully answer the query, acknowledge the limitations

At the end of your response, include TWO sections:
1. "Sources:" - List all sources used in this format: [Site Name] - Title (URL)
2. "Related Queries:" - Suggest 3-5 related questions the user might want to ask next
"""
    else:
        system_prompt = f"""
You are an advanced AI research assistant that provides comprehensive, factual, and balanced information.

The user has asked: "{user_query}"

Follow these principles:
1. Respond with in-depth, authoritative answers based ONLY on the provided sources
2. Start with a direct answer to the query, then explore details
3. Organize information with clear structure using headings and bullet points when appropriate
4. Highlight important facts, figures, and contrasting viewpoints
5. Maintain a balanced perspective, especially on controversial topics
6. Directly quote important statements or statistics from sources when relevant
7. Do NOT include information that isn't found in the provided sources
8. Ensure all facts are accurately tied to their sources
9. If the sources contradict each other, note the discrepancy
10. If sources are insufficient to fully answer the query, acknowledge the limitations

At the end of your response, include TWO sections:
1. "Sources:" - List all sources used in this format: [Site Name] - Title (URL)
2. "Related Queries:" - Suggest 3-5 related questions the user might want to ask next
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": combined_text}
    ]

    try:
        # Choose model based on complexity and token count
        model = "gpt-3.5-turbo"
        
        # Use GPT-4 for complex queries or when higher quality is needed
        complex_indicators = ["explain", "how", "why", "difference", "compare", "analysis", 
                             "implications", "future", "impact", "effects", "causes"]
        if any(word in user_query.lower() for word in complex_indicators) or len(user_query) > 80:
            model = "gpt-4-turbo-preview"
            max_tokens = max(max_tokens, 2000)
        
        # Log the model selection
        logger.info(f"Using model {model} with {max_tokens} max tokens")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,  # Lower temperature for more factual responses
            max_tokens=max_tokens
        )
        
        result = response.choices[0].message.content
        
        # Ensure source section is properly formatted
        if "Sources:" not in result:
            result += "\n\nSources:\n"
            for source in sources:
                result += f"- {source}\n"
        
        # Ensure related queries section is properly formatted
        if "Related Queries:" not in result:
            result += "\n\nRelated Queries:\n"
            # Generate related queries if they're missing
            try:
                related_prompt = f"Based on the query '{user_query}', generate 3-5 related follow-up questions a user might want to ask next. Make them concise and directly related to the topic."
                related_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": related_prompt}],
                    temperature=0.7,
                    max_tokens=150
                )
                related_queries = related_response.choices[0].message.content
                # Format the related queries
                for line in related_queries.split('\n'):
                    if line.strip() and not line.startswith('-'):
                        result += f"- {line.strip()}\n"
                    elif line.strip():
                        result += f"{line.strip()}\n"
            except Exception as e:
                # Fallback if we can't generate related queries
                logger.error(f"Error generating related queries: {str(e)}")
                result += "- What are the latest developments in this topic?\n"
                result += "- How does this compare to previous research?\n"
                result += "- What are the implications for the future?\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error with OpenAI API: {str(e)}")
        return f"Error processing query: {str(e)}"

# ===== MAIN ENGINE FUNCTION =====
def main(user_query):
    """Main function to process a user query and return results"""
    start_time = time.time()
    logger.info(f"Processing query: {user_query}")
    
    # Search for relevant links
    search_results = search_links(user_query)
    
    if not search_results:
        logger.warning("No search results found")
        return "I couldn't find any relevant information. Please try a different search query."
    
    # Extract text from search results concurrently
    articles = []
    max_workers = min(multiprocessing.cpu_count() * 2, 8)  # Limit concurrent requests
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for result in search_results:
            url = result.get('url')
            if url:
                future = executor.submit(fetch_text_from_url, url)
                futures.append(future)
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                article = future.result()
                if article and article.get('text') and len(article['text']) > 300:
                    articles.append(article)
            except Exception as e:
                logger.error(f"Error processing future: {str(e)}")
    
    if not articles:
        logger.warning("No valid articles found")
        return "I couldn't find relevant information. Please try a different search query."
    
    # Rank articles by relevance and quality
    logger.info(f"Found {len(articles)} valid articles")
    ranked_articles = rank_articles(articles, user_query)
    
    # Process with ChatGPT with appropriate token length
    tokens = 1500
    if len(user_query) > 100 or any(len(article['text']) > 5000 for article in articles):
        tokens = 2000
    
    final_result = process_with_chatgpt(user_query, ranked_articles, tokens)
    
    total_time = time.time() - start_time
    logger.info(f"Query processed in {total_time:.2f} seconds")
    
    return final_result

# ===== MODULE EXECUTION =====
if __name__ == "__main__":
    # Configure logging when running as main script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example test query
    result = main("What are the latest developments in quantum computing?")
    print(result)