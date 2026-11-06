import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pymupdf as fitz
from paddleocr import PaddleOCR

pdf = Path(sys.argv[1]).resolve()
scale = 2

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
)

fd, temporarer_pfad = tempfile.mkstemp(prefix=f".{pdf.stem}_ocr_", suffix=".pdf", dir=pdf.parent)
os.close(fd)
ausgabe = Path(temporarer_pfad)

try:
    with fitz.open(pdf) as original, fitz.open() as neu:
        for seite in original:
            # Nur das sichtbare Seitenbild übernehmen – die alte Textebene entfällt.
            pix = seite.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                colorspace=fitz.csRGB,
                alpha=False,
            )
            zielseite = neu.new_page(width=seite.rect.width, height=seite.rect.height)
            zielseite.insert_image(zielseite.rect, pixmap=pix)

            bild = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, 3
            ).copy()
            ergebnis = ocr.predict(bild)[0]

            # Nur neue, unsichtbare OCR-Textebene hinzufügen.
            for text, box in zip(ergebnis["rec_texts"], ergebnis["rec_boxes"]):
                x0, y0, x1, y1 = (float(wert) / scale for wert in box)
                breite, hoehe = x1 - x0, y1 - y0
                if not text.strip() or breite <= 0 or hoehe <= 0:
                    continue
                schriftgroesse = min(
                    hoehe * 0.8,
                    breite / max(fitz.get_text_length(text, fontsize=1), 0.1),
                )
                zielseite.insert_text(
                    (x0, y1 - hoehe * 0.15),
                    text,
                    fontsize=schriftgroesse,
                    render_mode=3,
                    overlay=True,
                )

        neu.save(ausgabe, deflate=True)

    # Erst nach erfolgreicher OCR die Original-PDF ersetzen.
    os.replace(ausgabe, pdf)
finally:
    # Bei Fehlern bleibt die Original-PDF erhalten.
    ausgabe.unlink(missing_ok=True)

print(f"Ersetzt: {pdf}")
