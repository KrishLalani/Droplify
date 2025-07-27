from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from urllib.parse import urlparse, urljoin
import re
from dotenv import load_dotenv
import statistics
import io
import threading

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- In-memory Database & Thread Lock ---
# Using a lock to prevent race conditions when multiple requests access the list
tracked_products = []
product_id_counter = 0
db_lock = threading.Lock()

class PriceTracker:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        self.smtp_config = {
            'email': os.getenv('SMTP_EMAIL'),
            'password': os.getenv('SMTP_PASSWORD'),
            'server': os.getenv('SMTP_SERVER'),
            'port': int(os.getenv('SMTP_PORT', 587))
        }

    def _get_clean_text(self, element):
        return re.sub(r'\s+', ' ', element.get_text()).strip() if element else ""

    def _extract_price_from_text(self, text):
        if not text: return None
        cleaned_text = re.sub(r'[₹,Rs.INR$A-Za-z\s]', '', str(text)).strip()
        match = re.search(r'(\d+\.\d*|\d+)', cleaned_text)
        if match:
            try:
                price = float(match.group(1))
                if 1 < price < 5000000: return price
            except (ValueError, TypeError): return None
        return None

    def _get_site_selectors(self, url):
        domain = urlparse(url).netloc.lower()
        # Generic selectors that work for many sites, with specific ones first
        selectors = {
            'title': ['#productTitle', '.B_NuCI', 'h1'],
            'price': ['.a-price-whole', '._30jeq3', '.a-price .a-offscreen', '#priceToPay'],
            'image': ['#landingImage', '._396cs4', '#imgTagWrapperId img'],
            'brand': ['#bylineInfo', '._2S4DIt a', '.po-brand .po-break-word'],
        }
        # Site-specific overrides can be added here if needed
        return selectors

    def scrape_product_info(self, url):
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            selectors = self._get_site_selectors(url)
            
            info = {'url': url, 'success': True, 'timestamp': datetime.now().isoformat()}

            for key, selector_list in selectors.items():
                for selector in selector_list:
                    element = soup.select_one(selector)
                    if element:
                        if key == 'price':
                            price = self._extract_price_from_text(element.get_text())
                            if price: info[key] = price; break
                        elif key == 'image':
                            src = element.get('src') or element.get('data-src')
                            if src: info['image_url'] = urljoin(url, src); break
                        else:
                            info[key] = self._get_clean_text(element)[:250]; break
            
            if 'title' not in info or 'price' not in info:
                raise ValueError("Could not extract Title or Price. The website layout may have changed.")

            return info

        except (requests.RequestException, ValueError, Exception) as e:
            print(f"Error scraping {url}: {e}")
            return {'success': False, 'error': str(e)}

    def send_email_alert(self, product, old_price):
        if not all(self.smtp_config.values()):
            print("🛑 SMTP Email configuration is incomplete. Skipping email alert.")
            return False
        
        subject = f"Price Drop for {product['title'][:40]}!"
        html_body = f"""
        <html><body style="font-family: sans-serif; margin: 0; padding: 20px; color: #333;">
        <div style="max-width: 600px; margin: auto; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
          <div style="background-color: #B6B09F; color: white; padding: 20px; text-align: center;">
            <h1 style="margin:0; font-size: 24px;">Price Alert from Droplify!</h1>
          </div>
          <div style="padding: 20px; text-align: center;">
            <p style="font-size: 16px;">The price for a product you're tracking has dropped!</p>
            <div style="margin: 20px 0;">
              <img src="{product.get('image_url', '')}" alt="Product Image" style="max-width: 200px; max-height: 200px; height: auto; border-radius: 4px; border: 1px solid #eee;">
            </div>
            <h2 style="font-size: 18px; margin: 10px 0;">{product['title']}</h2>
            <p style="font-size: 16px; margin: 5px 0;">Previous Price: <b style="text-decoration: line-through;">₹{old_price:,.2f}</b></p>
            <p style="font-size: 20px; margin: 5px 0;">New Price: <b style="color: #28a745;">₹{product['current_price']:,.2f}</b></p>
            <a href="{product['url']}" style="display: inline-block; margin-top: 20px; padding: 12px 25px; background-color: #B6B09F; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">
              Buy Now
            </a>
          </div>
          <div style="background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #6c757d;">
            <p>This alert was sent based on your threshold of "{product['threshold_value']}{'%' if product['threshold_type'] == 'percentage' else ' (fixed)'}".</p>
          </div>
        </div>
        </body></html>
        """
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['email']
            msg['To'] = product['email']
            msg['Subject'] = subject
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                server.starttls()
                server.login(self.smtp_config['email'], self.smtp_config['password'])
                server.send_message(msg)
                print(f"✅ Email sent to {product['email']} for product {product['id']}.")
            return True
        except smtplib.SMTPAuthenticationError as e:
            print(f"🛑 SMTP Authentication Error: Failed to send email. Check credentials and ensure you are using a Google App Password. Error: {e}")
            return False
        except Exception as e:
            print(f"🛑 Failed to send email: {e}")
            return False

price_tracker = PriceTracker()

def calculate_analytics(product):
    history = product.get('price_history', [])
    if not history: return {}
    return {
        'lowest_price': min(history),
        'highest_price': max(history),
        'average_price': round(statistics.mean(history), 2) if history else 0,
    }

# --- API Routes ---
@app.route('/')
def index_page():
    return render_template('index2.html')

@app.route('/api/add_product', methods=['POST'])
def add_product():
    global product_id_counter
    data = request.json
    url, email = data.get('url'), data.get('email')
    
    if not url or not email:
        return jsonify({'error': 'URL and Email are required.'}), 400

    with db_lock:
        if any(p for p in tracked_products if p['url'] == url and p['email'] == email):
            return jsonify({'error': 'You are already tracking this product.'}), 409

    info = price_tracker.scrape_product_info(url)
    if not info.get('success'):
        return jsonify({'error': f"Scraping failed: {info.get('error', 'Unknown error')}"}), 500
    
    if not isinstance(info.get('price'), (int, float)):
        return jsonify({'error': "Scraping failed: Could not determine a valid price for the product."}), 500

    with db_lock:
        product_id_counter += 1
        product = {
            'id': str(product_id_counter),
            'url': url,
            'email': email,
            'title': info.get('title', 'N/A'),
            'brand': info.get('brand'),
            'image_url': info.get('image_url'),
            'threshold_type': data['threshold_type'],
            'threshold_value': float(data['threshold_value']),
            'current_price': info['price'],
            'original_price': info['price'],
            'price_history': [info['price']],
            'timestamp_history': [datetime.now().isoformat()],
            'last_checked': datetime.now().isoformat(),
            'alerts_sent': 0,
        }
        tracked_products.append(product)
    
    return jsonify({'success': True, 'product': product})

@app.route('/api/products', methods=['GET'])
def get_products():
    with db_lock:
        products_with_analytics = []
        for p in tracked_products:
            new_p = p.copy()
            new_p['analytics'] = calculate_analytics(p)
            products_with_analytics.append(new_p)
    return jsonify({'products': products_with_analytics})

@app.route('/api/remove_product/<product_id>', methods=['DELETE'])
def remove_product(product_id):
    global tracked_products
    with db_lock:
        initial_len = len(tracked_products)
        tracked_products = [p for p in tracked_products if p['id'] != product_id]
        if len(tracked_products) == initial_len:
            return jsonify({'error': 'Product not found'}), 404
    return jsonify({'success': True})

# Other routes (check_prices, summary, greatest_deal, export) remain largely the same
# but are included here for completeness, wrapped with the db_lock.
@app.route('/api/check_prices', methods=['POST'])
def check_all_prices():
    alerts_sent_count = 0
    with db_lock:
        products_to_check = list(tracked_products)
    
    for product in products_to_check:
        info = price_tracker.scrape_product_info(product['url'])
        if info.get('success') and isinstance(info.get('price'), (int, float)):
            with db_lock:
                # Find the product again in the main list to update it
                p_to_update = next((p for p in tracked_products if p['id'] == product['id']), None)
                if not p_to_update: continue

                old_price = p_to_update['current_price']
                new_price = info['price']
                
                p_to_update['current_price'] = new_price
                p_to_update['last_checked'] = datetime.now().isoformat()
                p_to_update['price_history'].append(new_price)
                p_to_update['timestamp_history'].append(datetime.now().isoformat())
                
                alert_triggered = False
                if p_to_update['threshold_type'] == 'fixed' and new_price <= p_to_update['threshold_value']:
                    alert_triggered = True
                elif p_to_update['threshold_type'] == 'percentage':
                    drop = ((p_to_update['original_price'] - new_price) / p_to_update['original_price']) * 100
                    if drop >= p_to_update['threshold_value']:
                        alert_triggered = True
                
                if alert_triggered and new_price < old_price:
                    if price_tracker.send_email_alert(p_to_update, old_price):
                        p_to_update['alerts_sent'] += 1
                        alerts_sent_count += 1
                        
    return jsonify({'success': True, 'alerts_sent': alerts_sent_count})

@app.route('/api/analytics/summary')
def get_summary():
    with db_lock:
        total_savings = sum((p['original_price'] - p['current_price']) for p in tracked_products if p['original_price'] > p['current_price'])
        summary = {
            'total_products': len(tracked_products),
            'total_alerts_sent': sum(p.get('alerts_sent', 0) for p in tracked_products),
            'total_savings': round(total_savings, 2)
        }
    return jsonify({'success': True, 'summary': summary})

@app.route('/api/greatest_deal')
def get_greatest_deal():
    best_deal = None
    max_drop = -1
    with db_lock:
        products_copy = list(tracked_products)

    for product in products_copy:
        analytics = calculate_analytics(product)
        if analytics and 'highest_price' in analytics:
            highest = analytics['highest_price']
            current = product['current_price']
            if highest > current > 0:
                drop_percentage = ((highest - current) / highest) * 100
                if drop_percentage > max_drop:
                    max_drop = drop_percentage
                    best_deal = {
                        'title': product['title'], 'savings': round(highest - current, 2),
                        'drop_percentage': round(drop_percentage, 2), 'url': product['url']
                    }
    return jsonify({'success': True, 'deal': best_deal})

@app.route('/api/export_data')
def export_data():
    with db_lock:
        if not tracked_products:
            return jsonify({'error': 'No data to export.'}), 404
        data_str = json.dumps(tracked_products, indent=4)
    
    mem_file = io.BytesIO(data_str.encode('utf-8'))
    return send_file(mem_file, as_attachment=True,
        download_name=f'droplify_export_{datetime.now().strftime("%Y%m%d")}.json',
        mimetype='application/json')

if __name__ == '__main__':
    app.run(
        debug=os.getenv('DEBUG', 'False').lower() in ['true', '1', 't'],
        host=os.getenv('HOST', '0.0.0.0'), 
        port=int(os.getenv('PORT', 5001))
    )