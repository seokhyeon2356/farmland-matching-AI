
import easyocr
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import io

# --- FastAPI 앱 및 OCR 리더 초기화 ---

app = FastAPI(
    title="Re:Local OCR 서비스",
    description="이미지(토지등본 등)에서 텍스트를 추출하는 독립적인 AI 서비스입니다."
)

# 서버가 시작될 때 미리 한국어 모델을 메모리에 올려둡니다.
# 이렇게 하면 매번 요청이 올 때마다 모델을 불러오지 않아도 되므로 속도가 훨씬 빨라집니다.
try:
    print("EasyOCR 한국어 모델을 로딩합니다...")
    reader = easyocr.Reader(['ko'], gpu=False) # 한국어 모델 사용, CPU 모드
    print("EasyOCR 모델 로딩 완료.")
except Exception as e:
    print(f"EasyOCR 모델 로딩 중 오류 발생: {e}")
    reader = None

# --- API 엔드포인트 정의 ---

@app.get("/")
def read_root():
    return {"message": "Re:Local OCR 서비스에 오신 것을 환영합니다!"}

@app.post("/ocr/image-to-text", summary="이미지에서 텍스트 추출")
async def extract_text_from_image(file: UploadFile = File(...)):
    """
    사용자가 업로드한 이미지 파일에서 텍스트를 추출하여 반환합니다.

    - **file**: .png, .jpg, .jpeg 형식의 이미지 파일을 업로드하세요.
    """
    if reader is None:
        raise HTTPException(status_code=500, detail="OCR 모델이 로드되지 않았습니다. 서버 로그를 확인해주세요.")

    # 파일 확장자 확인
    allowed_extensions = {"png", "jpg", "jpeg"}
    extension = file.filename.split('.')[-1].lower()
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다. {allowed_extensions} 형식의 파일을 업로드해주세요.")

    try:
        # 업로드된 파일을 메모리에서 읽음
        contents = await file.read()
        image_stream = io.BytesIO(contents)

        # EasyOCR로 텍스트 추출
        # result는 [[bbox], "text", confidence] 형태의 리스트입니다.
        result = reader.readtext(image_stream.getvalue())

        # 추출된 텍스트만 리스트로 정리
        extracted_texts = [text for _, text, _ in result]

        if not extracted_texts:
            return {"extracted_text": [], "message": "이미지에서 텍스트를 추출하지 못했습니다."}

        return {"extracted_text": extracted_texts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 추출 중 오류가 발생했습니다: {e}")

