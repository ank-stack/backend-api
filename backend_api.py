from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import base64, io
from PIL import Image
import asyncio
from gradio_client import Client, handle_file

# Import your Gemini module
from qa_engine import get_answer_from_gemini

app = FastAPI(title="Exam Cheating Backend", version="1.0")

# ✅ Initialize PaddleOCR client
OCR_SPACE = "devilzzzz/PaddleOCR"   # your Hugging Face space name
ocr_client = Client(OCR_SPACE)


# Helper: run PaddleOCR
async def run_ocr_on_image(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    result = ocr_client.predict(
        img=handle_file(buf),
        api_name="/predict"
    )
    # result may be list/dict — normalize
    return str(result)


# Helper: extract question + options from OCR text
def parse_mcq_text(raw_text: str):
    """
    Very naive parser — adjust for your OCR pattern.
    Example input:
        '1. Who is the father of C language? a) Steve Jobs b) James Gosling c) Dennis Ritchie d) Rasmus Lerdorf'
    """
    text = raw_text.replace("\n", " ")
    if "?" in text:
        q_part, rest = text.split("?", 1)
        question = q_part.strip() + "?"
    else:
        question = text.strip()
        rest = ""

    options = []
    for opt in rest.split(")"):
        opt = opt.strip()
        if len(opt) > 1 and opt[0].lower() in ["a", "b", "c", "d"]:
            opt_text = opt[1:].strip("). ").strip()
            if opt_text:
                options.append(opt_text)

    return question, options


@app.post("/upload")
async def upload_image(request: Request):
    try:
        # 1️⃣ Parse JSON
        data = await request.json()
        image_data = data.get("image")
        if not image_data:
            return {"error": "No image provided"}

        # 2️⃣ Remove base64 prefix
        image_data = image_data.replace("data:image/png;base64,", "").replace("data:image/jpeg;base64,", "")

        # 3️⃣ Decode and save image
        image_bytes = base64.b64decode(image_data)
        tmp_path = "temp.png"
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)

        # 4️⃣ OCR using PaddleOCR (your Gradio Space)
        from gradio_client import Client, handle_file
        ocr_client = Client("devilzzzz/PaddleOCR")
        ocr_result = ocr_client.predict(img=handle_file(tmp_path), api_name="/predict")

        # 5️⃣ Parse OCR result
        ocr_text = " ".join([r[1][0] for r in ocr_result[0]]) if isinstance(ocr_result, list) else str(ocr_result)

        # 6️⃣ Pass question + options to Gemini engine
        from qa_engine import get_answer_from_gemini
        answer = get_answer_from_gemini(ocr_text, [])

        # 7️⃣ Return structured result
        return {
            "question_text": ocr_text,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend running"}