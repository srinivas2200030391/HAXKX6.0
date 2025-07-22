import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fpdf import FPDF

# Ensure "pdf" folder exists
os.makedirs("pdf", exist_ok=True)

# Set up WebDriver
options = Options()
options.add_argument("--headless")  
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open the website
url = "https://coin-shop-1vcx.onrender.com/category/ancient"
driver.get(url)

try:
    # Wait for elements to load
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "text-xl")))

    # Extract product names (filter out empty texts)
    products = [p.text.strip() for p in driver.find_elements(By.CLASS_NAME, "text-xl") if p.text.strip()]

    # Extract product prices (ensure prices contain ₹ symbol)
    prices = [p.text.strip() for p in driver.find_elements(By.CLASS_NAME, "text-3xl") if p.text.strip() and "₹" in p.text]

    # Ensure equal count of products and prices
    if len(products) != len(prices):
        print("⚠ Warning: Mismatch in product and price count!")
        print(f"Products found: {len(products)}, Prices found: {len(prices)}")

    # Replace problematic characters and format data
    def clean_text(text):
        return text.replace("₹", "is INR ").replace("–", "-").replace("—", "-")

    coin_data = [f" The price of {clean_text(name)}: {clean_text(price)}" for name, price in zip(products, prices)]

    # Save to PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)  # Default font (no need for extra fonts)

    pdf.cell(200, 10, "Bharat Coin Bazaar", ln=True, align='C')
    pdf.ln(10)

    for item in coin_data:
        pdf.cell(200, 10, item.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='L')

    pdf_path = "data/coin_prices.pdf"
    pdf.output(pdf_path, "F")

    print(f"✅ Data saved successfully in {pdf_path}")

except Exception as e:
    print("❌ Error:", e)

# Close the driver
driver.quit()
