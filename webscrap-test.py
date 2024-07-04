from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode (optional)

# Path to chromedriver
chrome_service = Service('/opt/homebrew/bin/chromedriver')

# Initialize the Chrome WebDriver
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# Open a website
driver.get('https://www.google.com')

# Print the title of the page
print(driver.title)

# Quit the driver
driver.quit()