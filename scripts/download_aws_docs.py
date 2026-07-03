import os
import requests
import json
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
except ImportError:
    # Simple progress indicator if tqdm is not installed
    tqdm = lambda x, **kwargs: x

# =====================================================================
# CONFIGURATION
# =====================================================================
# The root AWS documentation URL to crawl for cards
TARGET_URL = "https://docs.aws.amazon.com/vpc/"

# Base output directory where downloaded datasets will be stored
BASE_OUTPUT_DIR = "data/raw"

# Headers used for HTTP requests
USER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://docs.aws.amazon.com/"
}
# =====================================================================


def get_output_dir(url, base_output_path):
    """
    Extracts the relative path after the domain name and constructs the full output directory.
    E.g., "https://docs.aws.amazon.com/vpc/latest/userguide/" -> "data/raw/vpc/latest/userguide"
    """
    parsed = urlparse(url)
    relative_path = parsed.path.strip("/")
    return os.path.join(base_output_path, relative_path)


def resolve_meta_redirect(url, headers):
    """
    Fetches the page and follows HTML '<meta http-equiv="refresh" content="...URL=...">' redirects.
    Returns the final URL, response object, and BeautifulSoup page object.
    """
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
    if refresh:
        content = refresh.get('content', '')
        if isinstance(content, str) and 'URL=' in content:
            redirect_url = content.split('URL=')[-1].strip()
            final_url = urljoin(response.url, redirect_url)
            print(f"  -> Seguo re-indirizzamento HTML: {final_url}")
            response = requests.get(final_url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            return final_url, response, soup
            
    return response.url, response, soup


def extract_guide_links(contents, base_url):
    """
    Recursively extracts all page links from the TOC JSON structure,
    converting .html extensions to .md for direct markdown access.
    """
    links = []
    for item in contents:
        if "href" in item:
            href = item["href"]
            # Map .html links to their raw .md counterparts
            if href.endswith(".html"):
                href = href[:-5] + ".md"
            elif ".html#" in href:
                href = href.replace(".html#", ".md#")
            
            links.append({
                "title": item.get("title"),
                "url": urljoin(base_url, href),
                "filename": href.split("#")[0]  # remove anchor hash for saving
            })
        if "contents" in item:
            links.extend(extract_guide_links(item["contents"], base_url))
    return links


def download_markdown_file(url, file_path, headers):
    """
    Downloads a single markdown file if it does not already exist.
    """
    if os.path.exists(file_path):
        return True # Skip downloading if file already exists
        
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            return True
        else:
            print(f"\n[Errore] Impossibile scaricare {url}: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"\n[Errore] Eccezione per {url}: {e}")
        return False


def main():
    # 1. Scarica la pagina principale ed estrae le card
    print(f"Avvio crawler su: {TARGET_URL}")
    response = requests.get(TARGET_URL, headers=USER_HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    
    cards = []
    for card in soup.find_all('article', class_='awsdocs-card'):
        link_el = card.find('a', class_='awsdocs-card-link')
        title_el = card.find('h3', class_='awsdocs-card-title')
        if link_el and title_el:
            title = title_el.text.strip()
            href = link_el.get('href')
            if isinstance(href, str):
                absolute_card_url = urljoin(TARGET_URL, href)
                cards.append({
                    "title": title,
                    "url": absolute_card_url
                })
            
    print(f"Trovate {len(cards)} card.")
    
    # Crea la cartella principale per la pagina target
    root_output_dir = get_output_dir(TARGET_URL, BASE_OUTPUT_DIR)
    os.makedirs(root_output_dir, exist_ok=True)
    
    # Salva il file cards.json principale
    cards_json_path = os.path.join(root_output_dir, "cards.json")
    with open(cards_json_path, "w", encoding="utf-8") as f:
        json.dump(cards, f, indent=4, ensure_ascii=False)
    print(f"Indice delle card salvato in: {cards_json_path}")
    
    # 2. Elabora ciascuna card per scaricare il rispettivo dataset
    for card in cards:
        card_title = card["title"]
        card_url = card["url"]
        
        print(f"\nElaboro card: '{card_title}'")
        try:
            # Segui redirect HTML e ottieni la pagina reale
            resolved_url, _, card_soup = resolve_meta_redirect(card_url, USER_HEADERS)
            
            # Controlla se la pagina ha un sommario/TOC
            meta_tocs = card_soup.find('meta', attrs={'name': 'tocs'})
            if not meta_tocs:
                print(f"  -> Salto '{card_title}' (nessun file TOC trovato, es. API o CLI reference).")
                continue
                
            # Recupera il TOC
            tocs_file = meta_tocs.get('content')
            if not isinstance(tocs_file, str):
                print(f"  -> Salto '{card_title}' (sommario/TOC non valido).")
                continue
            toc_url = urljoin(resolved_url, tocs_file)
            
            # Base URL delle pagine della guida
            guide_base_url = urljoin(resolved_url, "./")
            
            # Crea la cartella di output per questa guida
            guide_output_dir = get_output_dir(guide_base_url, BASE_OUTPUT_DIR)
            os.makedirs(guide_output_dir, exist_ok=True)
            
            print(f"  -> Scarico TOC da: {toc_url}")
            toc_response = requests.get(toc_url, headers=USER_HEADERS)
            if toc_response.status_code != 200:
                print(f"  -> [Errore] Impossibile recuperare il sommario. Status: {toc_response.status_code}")
                continue
                
            toc_data = toc_response.json()
            all_guide_links = extract_guide_links(toc_data.get("contents", []), guide_base_url)
            print(f"  -> Trovati {len(all_guide_links)} file markdown.")
            
            # Salva il file guide_links.json locale
            guide_links_json_path = os.path.join(guide_output_dir, "guide_links.json")
            with open(guide_links_json_path, "w", encoding="utf-8") as f:
                saved_links = [
                    {"title": link["title"], "url": link["url"]}
                    for link in all_guide_links
                ]
                json.dump(saved_links, f, indent=4, ensure_ascii=False)
            print(f"  -> Guide JSON salvato in: {guide_links_json_path}")
            
            # Filtra solo i link che non sono ancora stati scaricati
            links_to_download = []
            for link in all_guide_links:
                file_url = link["url"]
                if "#" in file_url:
                    file_url = file_url.split("#")[0]
                file_path = os.path.join(guide_output_dir, link["filename"])
                if not os.path.exists(file_path):
                    links_to_download.append((file_url, file_path))
            
            if links_to_download:
                print(f"  -> Scarico {len(links_to_download)} nuovi file markdown (saltati {len(all_guide_links) - len(links_to_download)} già esistenti)...")
                for file_url, file_path in tqdm(links_to_download, desc=f"Downloading {card_title[:15]}"):
                    download_markdown_file(file_url, file_path, USER_HEADERS)
            else:
                print("  -> Tutti i file markdown di questa guida sono già stati scaricati.")
                
        except Exception as e:
            print(f"[Errore] Impossibile elaborare {card_url}: {e}")
            
    print("\n--- TUTTI I DOWNLOAD COMPLETATI! ---")


if __name__ == "__main__":
    main()
