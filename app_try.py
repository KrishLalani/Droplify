from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import requests, json, smtplib, time, threading, os, re, statistics, io, google.generativeai as genai
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from functools import wraps
from bs4 import BeautifulSoup

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- AI Configuration ---
try:
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    AI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
    AI_ENABLED = True
except Exception as e:
    AI_ENABLED = False
    print(f"Warning: Google AI could not be configured. AI features will be disabled. Error: {e}")


# --- In-memory Database & Persistence ---
DB_FILE = 'products.json'
tracked_products = []
product_id_counter = 0
db_lock = threading.Lock() # Lock for thread-safe DB operations

def load_products_from_disk():
    global tracked_products, product_id_counter
    with db_lock:
        if not os.path.exists(DB_FILE): return
        try:
            with open(DB_FILE, 'r') as f:
                tracked_products = json.load(f)
                if tracked_products:
                    valid_ids = [int(p['id']) for p in tracked_products if 'id' in p and str(p['id']).isdigit()]
                    product_id_counter = max(valid_ids) if valid_ids else 0
            print(f"Loaded {len(tracked_products)} products.")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading {DB_FILE}: {e}. Starting fresh.")
            tracked_products, product_id_counter = [], 0

def save_products_to_disk():
    with db_lock:
        try:
            with open(DB_FILE, 'w') as f:
                json.dump(tracked_products, f, indent=4)
        except IOError as e:
            print(f"Error saving to {DB_FILE}: {e}")

# --- Core Logic Class ---
class PriceTracker:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        })
        self.smtp_config = {k: os.getenv(f'SMTP_{k.upper()}') for k in ['email', 'password', 'server']}
        self.smtp_config['port'] = int(os.getenv('SMTP_PORT', 587))

    def _get_clean_text(self, element):
        return re.sub(r'\s+', ' ', element.get_text()).strip() if element else ""

    def _extract_price(self, text):
        if not text: return None
        cleaned = re.sub(r'[₹,Rs.INR$A-Za-z\s]', '', str(text)).strip()
        match = re.search(r'(\d+\.\d{1,2}|\d+)', cleaned)
        if match:
            price = float(match.group(1))
            return price if 10 < price < 5000000 else None
        return None
    
    def _find_first(self, soup, selectors, processor, attribute=None):
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                if attribute:
                    value = element.get(attribute)
                else:
                    value = processor(element) if processor else element
                if value: return value
        return None

    def _get_site_selectors(self, url):
        domain = urlparse(url).netloc.lower()
        if 'amazon' in domain:
            return {
                'title': ['#productTitle', 'h1#title'],
                'price': ['.a-price-whole', '.a-offscreen', '#corePrice_feature_div .a-price-whole'],
                'image': ['#landingImage', '#imgTagWrapperId img'], 'brand': ['#bylineInfo']
            }
        if 'flipkart' in domain:
            return {
                'title': ['.B_NuCI', 'h1._1_A1dD'], 'price': ['._30jeq3._16Jk6d', '._30jeq3'],
                'image': ['._396cs4._2amPTt._3qGmMb', 'img._396cs4'], 'brand': ['._2S4DIt a']
            }
        return { # Generic fallback
            'title': ['h1', '[itemprop="name"]'], 'price': ['[itemprop="price"]', '.price'],
            'image': ['[itemprop="image"]', '#landingImage'], 'brand': ['[itemprop="brand"]', '.brand-name'],
        }

    def scrape_product_info(self, url):
        try:
            if 'amazon' in urlparse(url).netloc:
                match = re.search(r'(dp|gp/product)/(\w+)', url)
                if match: url = f"https://www.amazon.in/dp/{match.group(2)}"

            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            selectors = self._get_site_selectors(url)

            title = self._find_first(soup, selectors['title'], self._get_clean_text) or self._get_clean_text(soup.find('title'))
            price = self._find_first(soup, selectors['price'], lambda el: self._extract_price(el.get_text()))
            image_src = self._find_first(soup, selectors['image'], None, 'src') or self._find_first(soup, selectors['image'], None, 'data-src')
            
            if not price:
                price_text = re.search(r'("price"|"amount")\s*:\s*"?(\d+\.?\d*)"?', response.text)
                if price_text: price = self._extract_price(price_text.group(2))
            
            if not title or not price:
                raise ValueError("Could not extract Title or Price. Website might be blocking scrapers or has a new layout.")

            return {'success': True, 'title': title[:200], 'price': price, 'url': url,
                    'image_url': urljoin(url, image_src) if image_src else None,
                    'brand': self._find_first(soup, selectors['brand'], self._get_clean_text),
                    'website': urlparse(url).netloc, 'timestamp': datetime.now().isoformat()}
        except (requests.RequestException, ValueError, Exception) as e:
            return {'success': False, 'error': str(e)}

    def _call_ai_model(self, prompt, error_msg="AI service is currently unavailable."):
        if not AI_ENABLED: return {"error": "AI service is not configured on the server."}
        try:
            response = AI_MODEL.generate_content(prompt)
            # Find the JSON block even if it's wrapped in markdown
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError("No valid JSON in AI response.")
        except Exception as e:
            print(f"Gemini API error: {e}")
            return {"error": error_msg}

    def get_ai_analysis(self, product_title, brand, current_price, price_history):
        prompt = f"""Analyze the product and return a valid JSON object only.
        Product: {product_title}, Brand: {brand or 'N/A'}, Current Price: ₹{current_price}, History: {price_history[-10:]}
        JSON structure: {{"description": "Brief product overview.", "price_analysis": "Analysis of current price vs history.", "recommendation": {{"decision": "BUY NOW" | "WAIT" | "CONSIDER", "reason": "Concise reason."}}, "buying_tip": "Actionable tip for this product type."}}"""
        return self._call_ai_model(prompt, "AI analysis is currently unavailable.")

    def get_ai_temporal_forecast(self, product_title, current_price, price_history):
        prompt = f"""You are a market analyst. Forecast the 30-day price trend for the product below. Your response must be a single, valid JSON object.
        Product: {product_title}, Current Price: ₹{current_price}, History: {price_history}
        JSON Structure: {{"forecast_summary": "One-sentence prediction.", "confidence": "Low" | "Medium" | "High", "reasoning": "Brief explanation (under 50 words).", "predicted_range": {{"low": <number>, "high": <number>}}}}"""
        return self._call_ai_model(prompt, "AI forecast is currently unavailable.")

    def get_ai_product_details(self, product_title, brand):
        prompt = f"""As a product expert, provide details for the item below. Return only a valid JSON object.
        Product Title: "{product_title}", Brand: "{brand or 'Not specified'}"
        JSON Structure: {{"summary": "Concise paragraph on the product's main purpose and key selling points.", "features": ["List of key features or benefits."], "specifications": {{"Key 1": "Value 1", "Key 2": "Value 2"}}}}"""
        return self._call_ai_model(prompt, "Could not fetch AI-powered product details.")

    def send_email_alert(self, product, old_price):
        if not all(self.smtp_config.values()): return False
        subject = f"🔥 Price Drop: {product['title'][:40]}!"
        html_body = f"""<html><body><h1>Price Drop Alert!</h1><h2>{product['title']}</h2>
                     <img src="{product.get('image_url', '')}" alt="Product" style="max-width: 200px;">
                     <p>Price dropped from <del>₹{old_price:,.2f}</del> to <strong>₹{product['current_price']:,.2f}</strong></p>
                     <a href="{product['url']}" style="padding: 10px; background-color: #5cb85c; color: white; text-decoration: none;">Buy Now</a>
                     </body></html>"""
        try:
            msg = MIMEMultipart()
            msg['From'], msg['To'], msg['Subject'] = self.smtp_config['email'], product['email'], subject
            msg.attach(MIMEText(html_body, 'html'))
            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                server.starttls(); server.login(self.smtp_config['email'], self.smtp_config['password']); server.send_message(msg)
            return True
        except Exception as e:
            print(f"Failed to send email: {e}"); return False

price_tracker = PriceTracker()

# --- Utility & Decorator ---
def calculate_analytics(product):
    history = product.get('price_history', [])
    if not history: return {}
    return {'lowest_price': min(history), 'highest_price': max(history), 'average_price': round(statistics.mean(history), 2),
            'price_change_percentage': round(((history[-1] - history[0]) / history[0]) * 100, 2) if len(history) > 1 and history[0] > 0 else 0}

def require_product(f):
    @wraps(f)
    def decorated_function(product_id, *args, **kwargs):
        product = next((p for p in tracked_products if str(p.get('id')) == str(product_id)), None)
        return f(product, *args, **kwargs) if product else (jsonify({'success': False, 'error': 'Product not found'}), 404)
    return decorated_function

# --- Background Scheduler ---
def scheduled_price_check():
    print(f"[{datetime.now()}] Running scheduled price check for {len(tracked_products)} products...")
    alerts_sent_count = 0
    with db_lock:
        for product in tracked_products:
            info = price_tracker.scrape_product_info(product['url'])
            if info.get('success'):
                old_price, new_price = product['current_price'], info['price']
                product['current_price'] = new_price
                product['last_checked'] = datetime.now().isoformat()
                product['price_history'].append(new_price)
                product['timestamp_history'].append(datetime.now().isoformat())
                
                drop = ((product['original_price'] - new_price) / product['original_price']) * 100 if product['original_price'] > 0 else 0
                alert_triggered = (product['threshold_type'] == 'fixed' and new_price <= product['threshold']) or \
                                  (product['threshold_type'] == 'percentage' and drop >= product['threshold'])
                
                if alert_triggered and new_price < old_price and price_tracker.send_email_alert(product, old_price):
                    product['alerts_sent'] += 1
                    alerts_sent_count += 1
            time.sleep(5) # Add a small delay between requests
    save_products_to_disk()
    print(f"Scheduled check finished. Alerts sent: {alerts_sent_count}")
    # Reschedule the next check
    threading.Timer(10800, scheduled_price_check).start() # 10800 seconds = 3 hours

# --- API Routes ---
@app.route('/')
def index_page():
    # Assumes the provided HTML is saved as 'templates/index.html'
    return render_template('index.html')

@app.route('/api/products', methods=['GET'])
def get_products():
    with db_lock:
        products_with_analytics = [dict(p, analytics=calculate_analytics(p)) for p in tracked_products]
    return jsonify({'products': products_with_analytics})

@app.route('/api/add_product', methods=['POST'])
def add_product():
    global product_id_counter
    data = request.json
    url, email = data.get('url'), data.get('email')
    if not url or not email: return jsonify({'success': False, 'error': 'URL and Email are required.'}), 400
    
    with db_lock:
        if any(p['url'] == url and p['email'] == email for p in tracked_products):
            return jsonify({'success': False, 'error': 'You are already tracking this product.'}), 409

    info = price_tracker.scrape_product_info(url)
    if not info.get('success'): return jsonify({'success': False, 'error': f"Scrape failed: {info.get('error', 'Unknown')}"}), 500
    
    with db_lock:
        product_id_counter += 1
        product = {
            'id': str(product_id_counter), 'url': url, 'email': email,
            'title': info['title'], 'brand': info.get('brand'), 'image_url': info.get('image_url'),
            'threshold_type': data.get('threshold_type', 'percentage'), 'threshold': float(data.get('targetValue', 10)),
            'current_price': info['price'], 'original_price': info['price'],
            'price_history': [info['price']], 'timestamp_history': [datetime.now().isoformat()],
            'last_checked': datetime.now().isoformat(), 'alerts_sent': 0, 'website': info.get('website'),
        }
        tracked_products.append(product)
    
    save_products_to_disk()
    return jsonify({'success': True, 'product': product})

@app.route('/api/product/<product_id>/edit', methods=['POST'])
@require_product
def edit_product(product):
    data = request.json
    try:
        new_threshold_type = data.get('threshold_type')
        new_threshold_value = float(data.get('targetValue'))
        
        with db_lock:
            product['threshold_type'] = new_threshold_type
            product['threshold'] = new_threshold_value
        
        save_products_to_disk()
        return jsonify({'success': True, 'message': 'Product updated successfully.'})
    except (ValueError, TypeError):
        return jsonify({'success': False, 'error': 'Invalid threshold value provided.'}), 400

@app.route('/api/remove_product/<product_id>', methods=['DELETE'])
@require_product
def remove_product(product):
    global tracked_products
    with db_lock:
        tracked_products = [p for p in tracked_products if p['id'] != product['id']]
    save_products_to_disk()
    return jsonify({'success': True})

@app.route('/api/product/<product_id>/insights', methods=['GET'])
@require_product
def get_product_insights_route(product):
    fresh_info = price_tracker.scrape_product_info(product['url'])
    current_price = fresh_info['price'] if fresh_info.get('success') else product['current_price']
    analysis = price_tracker.get_ai_analysis(product['title'], product.get('brand'), current_price, product['price_history'])
    if 'error' in analysis: return jsonify({'success': False, 'error': analysis['error']}), 500
    return jsonify({'success': True, 'analysis': analysis})

@app.route('/api/product/<product_id>/details', methods=['GET'])
@require_product
def get_product_details_route(product):
    details = price_tracker.get_ai_product_details(product['title'], product.get('brand'))
    if 'error' in details: return jsonify({'success': False, 'error': details['error']}), 500
    return jsonify({'success': True, 'details': details})

@app.route('/api/product/<product_id>/temporal_forecast', methods=['GET'])
@require_product
def get_temporal_forecast(product):
    history = product.get('price_history', [])
    if len(history) < 3: return jsonify({'success': False, 'error': 'Not enough price history for a reliable forecast.'}), 400
    
    fresh_info = price_tracker.scrape_product_info(product['url'])
    current_price = fresh_info['price'] if fresh_info.get('success') else product['current_price']
    
    forecast_data = price_tracker.get_ai_temporal_forecast(product['title'], current_price, history)
    if 'error' in forecast_data: return jsonify({'success': False, 'error': forecast_data['error']}), 500
    
    return jsonify({'success': True, 'product_title': product.get('title'),
                    'current_price': current_price, 'forecast': forecast_data})

@app.route('/api/check_prices', methods=['POST'])
def manual_check_all_prices():
    # This function is now just for the manual button click, the scheduled one runs automatically
    # We can run the check in a separate thread to avoid a long-hanging request
    threading.Thread(target=scheduled_price_check).start()
    return jsonify({'success': True, 'message': 'Price check initiated in the background. The list will update shortly.'})

@app.route('/api/export_data')
def export_data():
    if not tracked_products: return jsonify({'success': False, 'error': 'No data to export.'}), 404
    with db_lock:
        mem_file = io.BytesIO(json.dumps(tracked_products, indent=4).encode('utf-8'))
    return send_file(mem_file, as_attachment=True, download_name=f'droplify_export_{datetime.now():%Y%m%d}.json', mimetype='application/json')

@app.route('/api/greatest_deal', methods=['GET'])
def get_greatest_deal():
    best_deal = None; max_drop = -1
    with db_lock:
        for p in tracked_products:
            original, current = p.get('original_price'), p.get('current_price')
            if original and current and original > current:
                drop_percentage = ((original - current) / original) * 100
                if drop_percentage > max_drop:
                    max_drop = drop_percentage
                    best_deal = {'title': p.get('title'),'url': p.get('url'),
                                 'savings': round(original - current, 2),
                                 'drop_percentage': round(drop_percentage, 2)}
    return jsonify({'deal': best_deal})

# --- Main Execution ---
if __name__ == '__main__':
    load_products_from_disk()
    # Start the background scheduler for the first time
    threading.Timer(5, scheduled_price_check).start() # Start after 5s
    app.run(debug=True, host='0.0.0.0', port=5001)