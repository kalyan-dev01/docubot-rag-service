import logging
import re
import pymupdf as fitz

class PDFLoadError(Exception):
    pass

class PDFExtractionError(Exception):
    pass

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def detect_pdf_type(filepath:str,min_text_length:int = 20) -> str:
    try:
        doc = fitz.open(filepath)
        text = doc[0].get_text().strip()

        if len(text) > min_text_length:
            detected_type = 'Text'
            logger.info(f'Detected {filepath}:{detected_type}')
            return detected_type
        else:
            detected_type = 'Scanned'
            logger.info(f'Detected {filepath}:{detected_type}')
            return detected_type

    except Exception as e:
        logger.error('Error Loading PDF')
        raise PDFLoadError('Error Loading PDF') from e



def extract_pdf_pages(filepath:str)-> list[dict]:
    try:
        pages = []
        doc = fitz.open(filepath)
        logger.info(f'Loaded {len(doc)}')
        for page in range(len(doc)):
            try:
                text = doc[page].get_text()
                pages.append({
                    'page_number':page + 1,
                    'text':text
                })
            except Exception as e:
                logger.warning(f'Failed to parse page {page + 1}: {e}')
                continue

        logger.info(f'Extracted {len(pages)} pages' )

        return pages
        
    except Exception as e:
        logger.error('Failed to extract PDF pages')
        raise PDFExtractionError("Failed to extract") from e

def clean_page_text(text:str,header:str = "B. TECH: ACADEMIC REGULATIONS \nR-24 \n")-> str:
    if text.startswith(header):
        text = text[len(header):]

    text = re.sub(r"[ \t]+\n", "\n", text)

    return text

