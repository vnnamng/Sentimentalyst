from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
import time
import csv

def wait_and_get_element(by, selector):
    try:
        return wait.until(
            EC.presence_of_element_located((by, selector))
        )
    except TimeoutException:
        return

def wait_and_get_elements(by, selector):
    try:
        return wait.until(
            EC.presence_of_all_elements_located((by, selector))
        )
    except TimeoutException:
        return

def scrape_reviews(product_id):
    '''
    Scrapes first 10 results for each star rating of a product
    '''
    print(f"Scraping reviews for {product_id}")
    driver.get(f"http://www.amazon.com/product-reviews/{product_id}")
    
    with open("reviews.csv", "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for n in range(1, 6):
            star_filter = wait_and_get_element(By.XPATH,
                f"//a[contains(@class,'{n}star')]")
            if not star_filter:
                print(f"No {n} star reviews for {product_id}")
                continue
            action_chain(star_filter).click().perform()
            if n > 1 and reviews:
                wait.until(EC.all_of(*map(EC.staleness_of, reviews)))
            time.sleep(1)
            reviews = wait_and_get_elements(By.XPATH,
                "//div[contains(@id,'customer_review')]//span[@data-hook='review-body']")
            if not reviews:
                print(f"Error fetching {n} star reviews for {product_id}")
                continue
            for review in reviews:
                writer.writerow([review.text, n])

if __name__ == "__main__":
    # Set up driver
    chrome_device_manager = ChromeDriverManager().install()
    options = webdriver.ChromeOptions() 
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=options, service=Service(chrome_device_manager))
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'})
    wait = WebDriverWait(driver, 5)
    action_chain = ActionChains(driver).move_to_element

    # Initialise file
    with open("reviews.csv", "w", newline="") as csvfile:
        csv.writer(csvfile).writerow(["review", "stars"])

    # Get Bestsellers
    driver.get("https://www.amazon.com/Best-Sellers/zgbs")
    product_ids = wait_and_get_elements(By.XPATH,
        "//li[@class='a-carousel-card']/div")
    if not product_ids:
        time.sleep(3)
        driver.refresh()
        driver.get("https://www.amazon.com/Best-Sellers/zgbs")
        product_ids = wait_and_get_elements(By.XPATH,
            "//li[@class='a-carousel-card']/div")
    product_ids = [elem.get_attribute("data-asin") for elem in product_ids]
    print(f"{len(product_ids)} products found")

    for product_id in product_ids:
        scrape_reviews(product_id)

    driver.quit()
