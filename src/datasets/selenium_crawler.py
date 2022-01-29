
# Import
import json

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup


# Define Browser Options
from tqdm import tqdm

chrome_options = Options()
chrome_options.add_argument("--headless") # Hides the browser window
# Reference the local Chromedriver instance
chrome_path = r'/Users/sajad/Downloads/chromedriver'
driver = webdriver.Chrome(executable_path=chrome_path, options=chrome_options)
# Run the Webdriver, save page an quit browser

# driver.quit()



def _get_paper_url(id):
    return f'https://www.semanticscholar.org/paper/{id}'

def _run_selenium(url):
    driver.get(url)
    truncated_abstract = ''
    try:
        main_paper = driver.find_element(By.CSS_SELECTOR, "div#paper-header")
        truncated_abstract = main_paper.find_element(By.CLASS_NAME, "abstract__text").text
        expand_button = main_paper.find_element(By.CLASS_NAME, "cl-button__label")
        expand_button.click()
    except:
        return '', truncated_abstract.replace('Expand', '').strip()

    main_paper = driver.find_element(By.CSS_SELECTOR, "div#paper-header")
    title = main_paper.find_element(By.CSS_SELECTOR, "h1").text
    full_abstract = main_paper.find_element(By.CLASS_NAME, "abstract__text").text.replace('Collapse', '').strip()
    return title, full_abstract

if __name__ == '__main__':
    # BASE_DS_DIR = "/disk1/sajad/datasets/sci/arxivL/splits/"
    BASE_DS_DIR = ""
    SCHOLAR_BASE_PAGE = 'https://www.semanticscholar.org/paper/'
    for se in ['train']:
        paper_references = json.load(open(BASE_DS_DIR + f'reference_network_{se}_all.json', mode='r'))
        abstracts = {}

        for paper_id, references in tqdm(paper_references.items(), total=len(paper_references)):
            case_abstracts = []
            for ref in references:
                title, abstract = _run_selenium(_get_paper_url(ref['scholar_id']))
                if len(abstract) > 0:
                    case_abstracts.append({
                        'id': ref['scholar_id'],
                        'title': title,
                        'abstract': abstract
                    })
                    if len(case_abstracts) == 10:
                        break
            abstracts[paper_id] = case_abstracts


