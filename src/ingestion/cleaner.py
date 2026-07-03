"""
Modulo per la pulizia e la normalizzazione dei documenti grezzi.
Rimuove i tag HTML, normalizza i link e le immagini markdown,
e ripulisce la spaziatura in eccesso.
"""

import re
import logging

logger = logging.getLogger(__name__)

class DocumentCleaner:
    """
    Classe responsabile per la pulizia e normalizzazione del testo markdown.
    """

    def __init__(self, remove_html: bool = True, clean_links: bool = True, clean_images: bool = True):
        self.remove_html = remove_html
        self.clean_links = clean_links
        self.clean_images = clean_images

    def clean_markdown(self, text: str | None) -> str:
        """
        Pulisce il testo markdown applicando regex e normalizzazione di spaziatura.
        
        Args:
            text: Il testo markdown grezzo.
            
        Returns:
            Il testo pulito e normalizzato.
        """
        if not text:
            return ""

        cleaned = text

        # 1. Rimuove i tag HTML (es. ancore <a name="..."></a>)
        if self.remove_html:
            # Rimuove ancore HTML specifiche ed elementi vuoti
            cleaned = re.sub(r'<a\s+[^>]*>\s*</a>', '', cleaned, flags=re.IGNORECASE)
            # Rimuove tutti i restanti tag HTML generici
            cleaned = re.sub(r'<[^>]+>', '', cleaned)

        # 2. Pulisce le immagini markdown: ![alt](url) -> alt
        if self.clean_images:
            # Mantiene solo il testo alt dell'immagine per preservare il contesto semantico
            cleaned = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', cleaned)

        # 3. Pulisce i link markdown: [testo](url) -> testo
        if self.clean_links:
            # Rimpiazza i link con il solo testo del link
            cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)

        # 4. Normalizzazione degli spazi e dei ritorni a capo
        # Rimuove gli spazi di fine riga su ogni riga
        lines = [line.rstrip() for line in cleaned.splitlines()]
        cleaned = "\n".join(lines)
        
        # Sostituisce 3 o più ritorni a capo consecutivi con esattamente 2 (un paragrafo vuoto)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Rimuove spazi vuoti iniziali e finali del documento
        cleaned = cleaned.strip()

        return cleaned
