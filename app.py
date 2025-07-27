from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import requests, json, smtplib, time, threading, os, re, statistics, io, google.generativeai as genai, numpy as np, random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from functools import wraps
from bs4 import BeautifulSoup

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- AI Configuration ---
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
AI_MODEL = genai.GenerativeModel('gemini-1.5-flash')

# --- In-memory Database & Persistence ---
DB_FILE = 'products.json'
tracked_products = []
product_id_counter = 0

def load_products_from_disk():
    global tracked_products, product_id_counter
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
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1', 
            'Upgrade-Insecure-Requests': '1'
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
                'price': ['.a-price-whole', '.a-offscreen', '#corePrice_feature_div .a-price-whole', 'span[data-a-size="xl"] .a-offscreen'],
                'image': ['#landingImage', '#imgTagWrapperId img', '#main-image-container img'],
                'brand': ['#bylineInfo', 'tr.a-spacing-small td.a-span9 span']
            }
        if 'flipkart' in domain:
            return {
                'title': ['.B_NuCI', 'h1._1_A1dD'], 
                'price': ['._30jeq3._16Jk6d', '._30jeq3'], 
                'image': ['._396cs4._2amPTt._3qGmMb', 'img._396cs4', '._2r_T1I'], 
                'brand': ['._2S4DIt a', '.G6XhRU']
            }
        return { # Generic fallback
            'title': ['h1', '[itemprop="name"]', '#productTitle'],
            'price': ['[itemprop="price"]', '.price', '#price', '.a-price-whole', '._30jeq3'],
            'image': ['[itemprop="image"]', '#landingImage', '.product-image img'],
            'brand': ['[itemprop="brand"]', '.brand-name', '#bylineInfo'],
        }

    def scrape_product_info(self, url):
        try:
            # Clean Amazon URL
            if 'amazon' in urlparse(url).netloc:
                match = re.search(r'(dp|gp/product)/(\w+)', url)
                if match:
                    url = f"https://www.amazon.in/dp/{match.group(2)}"

            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            selectors = self._get_site_selectors(url)

            title = self._find_first(soup, selectors['title'], self._get_clean_text) or self._get_clean_text(soup.find('title'))
            price = self._find_first(soup, selectors['price'], lambda el: self._extract_price(el.get_text()))
            image_src = self._find_first(soup, selectors['image'], None, 'src') or self._find_first(soup, selectors['image'], None, 'data-src')
            
            if not price: # Fallback for price in JSON script
                price_text = re.search(r'("price"|"amount")\s*:\s*"?(\d+\.?\d*)"?', response.text)
                if price_text: price = self._extract_price(price_text.group(2))
            
            if not title or not price:
                raise ValueError("Could not extract essential product data (Title or Price). The website might be blocking scrapers or has changed its layout.")

            return {
                'success': True, 'title': title[:200], 'price': price, 'url': url,
                'image_url': urljoin(url, image_src) if image_src else None,
                'brand': self._find_first(soup, selectors['brand'], self._get_clean_text),
                'website': urlparse(url).netloc.replace('www.', ''), 'timestamp': datetime.now().isoformat()
            }
        except (requests.RequestException, ValueError, Exception) as e:
            print(f"Error scraping {url}: {e}")
            return {'success': False, 'error': str(e)}

    def search_product_on_multiple_sites(self, product_title, brand=None, original_url=None):
        clean_title = re.sub(r'\([^)]*\)', '', product_title).strip()
        query = f"{brand} {clean_title} price" if brand else f"{clean_title} price"
        google_url = f"https://www.google.com/search?q={requests.utils.quote(query)}&tbm=shop"
        print(f"Comparison search: '{query}'")
        try:
            response = self.session.get(google_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            results, seen = [], {urlparse(original_url).netloc.replace('www.', '')} if original_url else set()
            for item in soup.select('.sh-dgr__content'):
                website = self._get_clean_text(item.select_one('.IuHnof')).lower().replace('...','')
                if not website or website in seen: continue
                
                price = self._extract_price(self._get_clean_text(item.select_one('.a8Pemb')))
                link_elem = item.select_one('a.sh-dgr__offer-content')

                if price and link_elem:
                    results.append({'website': website, 'price': price, 'url': urljoin(google_url, link_elem['href'])})
                    seen.add(website)
                if len(results) >= 5: break
            return sorted(results, key=lambda x: x['price'])
        except Exception as e:
            print(f"Error in comparison search: {e}")
            return []

    def get_ai_analysis(self, product_title, brand, current_price, price_history):
        try:
            prompt = f"""
            Analyze the product and return a valid JSON object only. Do not include any text before or after the JSON object.
            Product Data:
            - Title: {product_title}
            - Brand: {brand or 'Unknown'}
            - Current Price: ₹{current_price}
            - Recent Price History (oldest to newest): {price_history[-10:]}

            First, infer the product's general category (e.g., 'Smartphone', 'Running Shoes', 'Kitchen Appliance').
            Then, based on the category and price data, provide your analysis.

            JSON structure: {{
              "description": "Brief product description, inferring details from the title. Should be 1-2 sentences.",
              "price_analysis": "Analysis of current price (₹{current_price}) vs. its history. Mention lowest/average price and comment on the trend.",
              "recommendation": {{"decision": "BUY NOW" | "WAIT" | "CONSIDER", "reason": "Concise reason, considering the product category and price trend."}},
              "buying_tip": "A general, actionable tip for buying this type of product."
            }}
            Generate the JSON response.
            """
            response = AI_MODEL.generate_content(prompt)
            # Find the JSON blob, even if it's wrapped in markdown
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not json_match:
                raise ValueError("AI response did not contain a JSON object.")
            return json.loads(json_match.group(0))
        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Gemini API error or JSON parsing failed (get_ai_analysis): {e}")
            return {"error": "AI analysis is currently unavailable."}
            
    # NEW METHOD: For product comparison
    def get_ai_comparison(self, products_data):
        """Generates AI comparison with better error handling"""
        if not products_data or len(products_data) < 2:
            return {"error": "At least two products are required for comparison."}
        
        try:
            product_summaries = []
            for i, product in enumerate(products_data):
                summary = f"""Product {i+1}:
- Title: {product.get('title', 'Unknown')}
- Price: ₹{product.get('price', 'N/A')}
- Brand: {product.get('brand', 'Unknown')}
- Website: {product.get('website', 'Unknown')}"""
                product_summaries.append(summary)
            
            # Simple feature inference for better comparison
            all_brands = [p.get('brand', 'Unknown') for p in products_data]
            all_prices = [f"₹{p.get('price', 'N/A')}" for p in products_data]

            prompt = f"""You are a product comparison expert. Analyze the following products based on their titles and prices.
Provide your response in a valid JSON format ONLY, with no other text before or after the JSON object.

Products to Compare:
{chr(10).join(product_summaries)}

Based on the titles, infer a few key comparison features (e.g., for 'Apple iPhone 15 Pro Max 256GB', you can infer Storage: '256GB').
Create a concise comparison table and determine the best value product.

Use this exact JSON structure:
{{
  "comparison_summary": "A brief, one or two-sentence summary comparing the key differences and strengths of the products.",
  "feature_comparison": {{
    "Price": {json.dumps(all_prices)},
    "Brand": {json.dumps(all_brands)}
  }},
  "best_value": {{
    "product_index": 0,
    "reason": "A short, clear explanation of why this product is the best value. Consider its price relative to its likely features."
  }}
}}

Instructions for AI:
1.  In 'feature_comparison', add 2-4 more relevant features you can infer from the product titles (e.g., 'Storage', 'Screen Size', 'Model'). If you can't infer a feature for a product, use 'N/A'. The values for each feature must be a list of strings, one for each product.
2.  The 'product_index' for 'best_value' must be the 0-based index of the winning product from the list I provided.
3.  Choose the 'best_value' product by balancing its features (inferred from title) against its price.
"""
            
            response = AI_MODEL.generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                raise ValueError("No valid JSON found in AI response.")
        
        except Exception as e:
            print(f"Error in get_ai_comparison: {e}")
            return {"error": f"AI comparison failed: {str(e)}"}

    def get_ai_temporal_forecast(self, product_title, current_price, price_history):
        """Generates a qualitative, AI-driven price forecast."""
        try:
            prompt = f"""
            You are a market analyst AI specializing in e-commerce price trends. Based on the following product data, provide a price forecast for the next 30 days.
            Your response must be a single, valid JSON object and nothing else.

            Product Title: {product_title}
            Current Price: ₹{current_price}
            Recent Price History (oldest to newest): {price_history}

            JSON structure:
            {{
              "forecast_summary": "A one-sentence summary of your prediction (e.g., 'Price likely to drop slightly').",
              "confidence": "Low" | "Medium" | "High",
              "reasoning": "A brief explanation for your forecast. Mention factors like product type (e.g., electronics, fashion), seasonality, and the observed price trend. Keep it under 50 words.",
              "predicted_range": {{
                "low": <number>,
                "high": <number>
              }}
            }}

            Guidelines:
            - Base your 'confidence' on the amount and volatility of the historical data. More data means higher confidence.
            - Calculate the 'predicted_range' based on the data. The 'low' should not be more than 20% below the lowest historical price, and the 'high' should not be more than 10% above the highest historical price to keep it realistic.
            """
            response = AI_MODEL.generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not json_match:
                raise ValueError("AI response did not contain a JSON object.")
            return json.loads(json_match.group(0))
        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Gemini API error or JSON parsing failed (get_ai_temporal_forecast): {e}")
            return {"error": "AI forecast is currently unavailable."}

    def send_email_alert(self, product, old_price):
        if not all(self.smtp_config.values()):
            print("Email config incomplete. Skipping alert.")
            return False
        
        try: 
            recommendation_prompt = f"Product '{product['title']}' dropped from ₹{old_price} to ₹{product['current_price']}. Recommend 'BUY NOW' or 'WAIT' with a very short reason."
            recommendation = AI_MODEL.generate_content(recommendation_prompt).text.strip()
        except Exception as e:
            print(f"Gemini API error (email alert): {e}")
            recommendation = "Check the deal now!"

        subject = f"🔥 Price Drop: {product['title'][:40]}!"
        html_body = f"""
        <html><body>
        <h1 style="color: #d9534f;">Price Drop Alert!</h1>
        <h2>{product['title']}</h2>
        <img src="{product.get('image_url', '')}" alt="Product" style="max-width: 200px;">
        <p>Price dropped from <del>₹{old_price:,.2f}</del> to <strong>₹{product['current_price']:,.2f}</strong></p>
        <p><strong>AI Recommendation:</strong> {recommendation}</p>
        <a href="{product['url']}" style="padding: 10px; background-color: #5cb85c; color: white; text-decoration: none;">Buy Now</a>
        </body></html>
        """
        try:
            msg = MIMEMultipart()
            msg['From'], msg['To'], msg['Subject'] = self.smtp_config['email'], product['email'], subject
            msg.attach(MIMEText(html_body, 'html'))
            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                server.starttls()
                server.login(self.smtp_config['email'], self.smtp_config['password'])
                server.send_message(msg)
                print(f"Email sent to {product['email']} for product {product['id']}.")
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

price_tracker = PriceTracker()

# --- Utility & Decorator ---
def calculate_analytics(product):
    history = product.get('price_history', [])
    if not history: return {}
    return {
        'lowest_price': min(history), 'highest_price': max(history),
        'average_price': round(statistics.mean(history), 2),
        'price_change_percentage': round(((history[-1] - history[0]) / history[0]) * 100, 2) if len(history) > 1 and history[0] > 0 else 0
    }

def require_product(f):
    @wraps(f)
    def decorated_function(product_id, *args, **kwargs):
        product = next((p for p in tracked_products if str(p.get('id')) == str(product_id)), None)
        return f(product, *args, **kwargs) if product else jsonify({'success': False, 'error': 'Product not found'}), 404
    return decorated_function


def get_product_details_ai(product_title, brand=None):
    """Fetches detailed product specifications and features using the AI model."""
    try:
        prompt = f"""
        As a product details expert, analyze the following product and provide a structured JSON response.
        Do not include any text before or after the JSON object.

        Product Title: "{product_title}"
        Brand: "{brand or 'Not specified'}"

        Return a single JSON object with the following schema:
        {{
          "summary": "A concise paragraph summarizing the product's main purpose and key selling points.",
          "features": [
            "A key feature or benefit as a string.",
            "Another key feature or benefit."
          ],
          "specifications": {{
            "Key 1": "Value 1",
            "Key 2": "Value 2"
          }}
        }}

        If you cannot find specific details, return null for the corresponding value but maintain the overall JSON structure. For example, if no features are found, return "features": [].
        """
        response = AI_MODEL.generate_content(prompt)
        # Find the JSON blob, even if it's wrapped in markdown
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_match:
            raise ValueError("AI response did not contain a JSON object.")
        return json.loads(json_match.group(0))
    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"Gemini API error or JSON parsing failed (get_product_details_ai): {e}")
        return None

# --- [NEW] HOT DEALS LOGIC ---
def get_mock_deals():
    """
    Mock function to simulate Amazon hot deals
    """
    deals = [
        {
            "id": 1,
            "title": "Apple AirPods Pro (2nd Generation)",
            "original_price": 24900.00,
            "sale_price": 17990.00,
            "discount_percentage": 28,
            "image_url": "https://via.placeholder.com/300x300?text=AirPods+Pro",
            "product_url": "#",
            "rating": 4.8,
            "reviews_count": 52847,
            "deal_ends": "2025-07-30T23:59:59"
        },
        {
            "id": 2,
            "title": "Samsung 65-inch Crystal 4K UHD Smart TV",
            "original_price": 89990.00,
            "sale_price": 59990.00,
            "discount_percentage": 33,
            "image_url": "https://via.placeholder.com/300x300?text=Samsung+TV",
            "product_url": "#",
            "rating": 4.5,
            "reviews_count": 12456,
            "deal_ends": "2025-07-29T18:00:00"
        },
        {
            "id": 3,
            "title": "Instant Pot Duo 7-in-1 Electric Pressure Cooker",
            "original_price": 9999.00,
            "sale_price": 5999.00,
            "discount_percentage": 40,
            "image_url": "https://via.placeholder.com/300x300?text=Instant+Pot",
            "product_url": "#",
            "rating": 4.7,
            "reviews_count": 89234,
            "deal_ends": "2025-07-28T12:00:00"
        },
        {
            "id": 4,
            "title": "Echo Dot (5th Gen) Smart Speaker with Alexa",
            "original_price": 4999.00,
            "sale_price": 2499.00,
            "discount_percentage": 50,
            "image_url": "https://via.placeholder.com/300x300?text=Echo+Dot",
            "product_url": "#",
            "rating": 4.6,
            "reviews_count": 67890,
            "deal_ends": "2025-07-31T23:59:59"
        },
        {
            "id": 5,
            "title": "Nike Men's Air Max 270 Running Shoes",
            "original_price": 15000.00,
            "sale_price": 8999.00,
            "discount_percentage": 40,
            "image_url": "https://via.placeholder.com/300x300?text=Nike+Shoes",
            "product_url": "#",
            "rating": 4.4,
            "reviews_count": 23456,
            "deal_ends": "2025-07-29T23:59:59"
        },
        {
            "id": 6,
            "title": "boAt Airdopes 141 Bluetooth TWS Earbuds",
            "original_price": 4490.00,
            "sale_price": 999.00,
            "discount_percentage": 78,
            "image_url": "https://via.placeholder.com/300x300?text=boAt+Airdopes",
            "product_url": "#",
            "rating": 4.1,
            "reviews_count": 175000,
            "deal_ends": "2025-07-28T23:59:59"
        }
    ]
    return deals

# --- API Routes ---
@app.route('/')
def index_page():
    return render_template('index.html')

# [NEW] Hot Deals Route
@app.route('/api/hot-deals')
def get_hot_deals():
    try:
        deals = get_mock_deals()
        return jsonify({
            "status": "success",
            "data": deals,
            "timestamp": datetime.now().isoformat(),
            "total_deals": len(deals)
        })
    except Exception as e:
        return jsonify({"status": "error","message": str(e),"data": []}), 500

# [NEW] Single Deal Route
@app.route('/api/deal/<int:deal_id>')
def get_deal_details(deal_id):
    try:
        deals = get_mock_deals()
        deal = next((d for d in deals if d["id"] == deal_id), None)
        if deal:
            return jsonify({"status": "success","data": deal})
        else:
            return jsonify({"status": "error", "message": "Deal not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# [NEW] Filter by Category Route
@app.route('/api/deals/category/<category>')
def get_deals_by_category(category):
    try:
        all_deals = get_mock_deals()
        category_keywords = {
            'electronics': ['apple', 'samsung', 'echo', 'tv', 'smart', 'airdopes', 'earbuds'],
            'home': ['instant pot', 'kitchen', 'home'],
            'fashion': ['nike', 'shoes', 'clothing'],
            'sports': ['nike', 'sports', 'fitness']
        }
        
        keywords = category_keywords.get(category.lower(), [])
        if not keywords:
             return jsonify({ "status": "success", "data": all_deals, "category": category, "total_deals": len(all_deals)})

        filtered_deals = []
        for deal in all_deals:
            for keyword in keywords:
                if keyword.lower() in deal['title'].lower():
                    if deal not in filtered_deals:
                        filtered_deals.append(deal)
        
        return jsonify({
            "status": "success",
            "data": filtered_deals,
            "category": category,
            "total_deals": len(filtered_deals)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Existing Droplify API Routes ---
@app.route('/api/products', methods=['GET'])
def get_products():
    products_with_analytics = []
    for p in tracked_products:
        p_copy = p.copy()
        p_copy['analytics'] = calculate_analytics(p_copy)
        products_with_analytics.append(p_copy)
    return jsonify({'products': products_with_analytics})

@app.route('/api/add_product', methods=['POST'])
def add_product():
    global product_id_counter
    data = request.json
    url, email = data.get('url'), data.get('email')
    if not url or not email:
        return jsonify({'success': False, 'error': 'URL and Email are required.'}), 400
    if any(p['url'] == url and p['email'] == email for p in tracked_products):
        return jsonify({'success': False, 'error': 'You are already tracking this product.'}), 409

    info = price_tracker.scrape_product_info(url)
    if not info.get('success'):
        return jsonify({'success': False, 'error': f"Scrape failed: {info.get('error', 'Unknown')}"}), 500
    
    product_details = get_product_details_ai(info['title'], info.get('brand'))

    product_id_counter += 1
    product = {
        'id': str(product_id_counter), 'url': url, 'email': email,
        'title': info['title'], 'brand': info.get('brand'), 'image_url': info.get('image_url'),
        'threshold_type': data.get('threshold_type', 'percentage'), 'threshold': float(data.get('threshold_value', 10)),
        'current_price': info['price'], 'original_price': info['price'],
        'price_history': [info['price']], 'timestamp_history': [datetime.now().isoformat()],
        'last_checked': datetime.now().isoformat(), 'alerts_sent': 0, 'website': info.get('website'),
        'price_comparison': price_tracker.search_product_on_multiple_sites(info['title'], info.get('brand'), url),
        'details': product_details
    }
    tracked_products.append(product)
    save_products_to_disk()
    return jsonify({'success': True, 'product': product})


@app.route('/api/remove_product/<product_id>', methods=['DELETE'])
@require_product
def remove_product(product):
    global tracked_products
    tracked_products = [p for p in tracked_products if p['id'] != product['id']]
    save_products_to_disk()
    return jsonify({'success': True})


@app.route('/api/product/<product_id>/insights', methods=['GET'])
@require_product
def get_product_insights_route(product):
    try:
        fresh_info = price_tracker.scrape_product_info(product['url'])
        current_price = fresh_info['price'] if fresh_info.get('success') else product['current_price']
        
        price_comparison = price_tracker.search_product_on_multiple_sites(
            product['title'], product.get('brand'), product['url']
        )
        
        analysis = price_tracker.get_ai_analysis(
            product['title'], product.get('brand'), current_price, product['price_history']
        )

        return jsonify({
            'success': True,
            'analysis': analysis,
            'price_comparison': price_comparison
        })
    except Exception as e:
        print(f"Error in /insights route: {e}")
        return jsonify({'success': False, 'error': 'An unexpected server error occurred.'}), 500

@app.route('/api/quick_action', methods=['POST'])
def quick_action():
    data = request.json
    url, action = data.get('url'), data.get('action')

    if not url or not action:
        return jsonify({'success': False, 'error': 'URL and action are required.'}), 400

    info = price_tracker.scrape_product_info(url)
    if not info.get('success'):
        return jsonify({'success': False, 'error': f"Scraping failed: {info.get('error', 'Could not retrieve product data from the URL.')}"}), 500

    if action == 'recommendation':
        analysis = price_tracker.get_ai_analysis(info['title'], info.get('brand'), info['price'], [info['price']])
        if 'error' in analysis:
            return jsonify({'success': False, 'error': analysis['error']}), 500
        return jsonify({'success': True, 'analysis': {'recommendation': analysis.get('recommendation')}})

    elif action == 'insights':
        analysis = price_tracker.get_ai_analysis(info['title'], info.get('brand'), info['price'], [info['price']])
        comparison = price_tracker.search_product_on_multiple_sites(info['title'], info.get('brand'), url)
        if 'error' in analysis:
            return jsonify({'success': False, 'error': analysis['error']}), 500
        return jsonify({
            'success': True,
            'analysis': analysis,
            'price_comparison': comparison
        })

    elif action == 'prediction':
        return jsonify({'success': False, 'error': 'A reliable forecast requires more price history. Please track the product first.'}), 400
    
    elif action == 'history':
        return jsonify({'success': True, 'current_price': info['price']})
    
    else:
        return jsonify({'success': False, 'error': 'Invalid action specified.'}), 400


@app.route('/api/check_prices', methods=['POST'])
def check_all_prices():
    alerts_sent_count = 0
    for product in tracked_products:
        info = price_tracker.scrape_product_info(product['url'])
        if info.get('success'):
            old_price, new_price = product['current_price'], info['price']
            
            product['current_price'] = new_price
            product['last_checked'] = datetime.now().isoformat()
            product['price_history'].append(new_price)
            product['timestamp_history'].append(datetime.now().isoformat())
            
            drop = ((product['original_price'] - new_price) / product['original_price']) * 100 if product['original_price'] > 0 else 0
            alert_triggered = (product.get('threshold_type') == 'fixed' and new_price <= product.get('threshold')) or \
                              (product.get('threshold_type') == 'percentage' and drop >= product.get('threshold'))
            
            if alert_triggered and new_price < old_price:
                if price_tracker.send_email_alert(product, old_price):
                    product['alerts_sent'] = product.get('alerts_sent', 0) + 1
                    alerts_sent_count += 1
    
    save_products_to_disk()
    return jsonify({'success': True, 'alerts_sent': alerts_sent_count})

@app.route('/api/export_data')
def export_data():
    if not tracked_products:
        return jsonify({'success': False, 'error': 'No data to export.'}), 404
    mem_file = io.BytesIO(json.dumps(tracked_products, indent=4).encode('utf-8'))
    return send_file(mem_file, as_attachment=True, download_name=f'droplify_export_{datetime.now():%Y%m%d}.json', mimetype='application/json')

@app.route('/api/product/<product_id>/temporal_forecast')
@require_product
def get_temporal_forecast(product):
    history = product.get('price_history', [])
    if len(history) < 3:
        return jsonify({'success': False, 'error': 'Not enough price history for a reliable forecast.'}), 400

    fresh_info = price_tracker.scrape_product_info(product['url'])
    current_price = fresh_info['price'] if fresh_info.get('success') else product['current_price']

    forecast_data = price_tracker.get_ai_temporal_forecast(
        product['title'], current_price, history
    )
    
    if 'error' in forecast_data:
        return jsonify({'success': False, 'error': forecast_data['error']}), 500
        
    return jsonify({
        'success': True,
        'product_title': product.get('title'),
        'current_price': current_price,
        'forecast': forecast_data
    })


@app.route('/api/greatest_deal', methods=['GET'])
def get_greatest_deal():
    best_deal = None
    max_drop = -1

    for p in tracked_products:
        original = p.get('original_price')
        current = p.get('current_price')
        if original and current and original > 0 and original > current:
            drop_percentage = ((original - current) / original) * 100
            if drop_percentage > max_drop:
                max_drop = drop_percentage
                best_deal = {
                    'title': p.get('title'),
                    'url': p.get('url'),
                    'savings': round(original - current, 2),
                    'drop_percentage': round(drop_percentage, 2)
                }

    return jsonify({'deal': best_deal}) if best_deal else jsonify({'deal': None})

# NEW ROUTE: For product comparison
@app.route('/api/compare_products', methods=['POST'])
def compare_products_route():
    try:
        urls = request.json.get('urls', [])
        if len(urls) < 2:
            return jsonify({
                'success': False, 
                'error': 'Please provide at least two URLs for comparison.'
            }), 400
        
        scraped_products = []
        for url in urls:
            if url.strip():
                result = price_tracker.scrape_product_info(url)
                if result.get('success'):
                    scraped_products.append(result)
        
        if len(scraped_products) < 2:
            return jsonify({
                'success': False,
                'error': 'Could not scrape enough valid product pages for comparison. Please check the URLs.'
            }), 400
        
        ai_analysis = price_tracker.get_ai_comparison(scraped_products)
        if 'error' in ai_analysis:
            return jsonify({
                'success': False,
                'error': ai_analysis['error']
            }), 500
        
        return jsonify({
            'success': True,
            'scraped_data': scraped_products,
            'ai_analysis': ai_analysis
        })
        
    except Exception as e:
        print(f"Error in /api/compare_products: {e}")
        return jsonify({
            'success': False,
            'error': f'An unexpected server error occurred: {str(e)}'
        }), 500


# --- Main Execution ---
if __name__ == '__main__':
    load_products_from_disk()
    app.run(debug=True, host='0.0.0.0', port=5001)