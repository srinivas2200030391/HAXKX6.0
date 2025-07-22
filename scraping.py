import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fpdf import FPDF

# Create "pdf" folder if it doesn't exist
if not os.path.exists("pdf"):
    os.makedirs("pdf")

# Set up WebDriver
options = Options()
options.add_argument("--headless")  # Run in headless mode
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open the website
url = "https://coin-shop-1vcx.onrender.com/category/ancient"
driver.get(url)

# Wait for products to load
try:
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "text-xl")))
    
    products = driver.find_elements(By.CLASS_NAME, "text-xl")  # Coin names
    prices = driver.find_elements(By.CLASS_NAME, "text-3xl")  # Prices

    # Prepare data
    coin_data = []
    for name, price in zip(products, prices):
        coin_data.append(f"{name.text}: {price.text}")

    # Save to PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("helvetica", "", "C:/Windows/Fonts/arial.ttf", uni=True)  # Load a Unicode font
    pdf.set_font("helvetica", size=12)

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
