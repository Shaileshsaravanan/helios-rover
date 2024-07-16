from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time

# Set up Chrome options to automatically accept geolocation permissions
options = webdriver.ChromeOptions()
options.add_experimental_option("prefs", {
    "profile.default_content_setting_values.geolocation": 1
})

# Set up the Chrome WebDriver
service = Service(executable_path='/opt/homebrew/bin/chromedriver')  # Update with the path to your chromedriver
driver = webdriver.Chrome(service=service, options=options)

try:
    # Load the webpage
    driver.get("test.html")  # Update with the path to your HTML file

    # Wait for the page to load and the button to be available
    wait = WebDriverWait(driver, 10)
    get_location_button = wait.until(EC.element_to_be_clickable((By.ID, "getLocationButton")))

    # Click the button to get the location
    get_location_button.click()

    # Keep clicking the button at intervals
    while True:
        time.sleep(1)
        get_location_button.click()
finally:
    # Clean up by closing the browser after some time
    time.sleep(0.1)  # Keep the browser open for a while to observe the actions
    driver.quit()