import pytesseract
import cv2  # OpenCV로 이미지 전처리 (정확도 향상)

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Windows 예시

def extract_text_with_tesseract(image_path, lang='kor'): # kor+eng
    """
    Tesseract를 사용해 이미지에서 텍스트 추출
    lang: 'eng' (영어), 'kor' (한국어), 'kor+eng' (한글+영어 혼합 추천)
    """
    # 이미지 읽기
    img = cv2.imread(image_path)

    # 전처리: 그레이스케일 + 이진화 (정확도 크게 향상)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Tesseract로 텍스트 추출
    extracted_text = pytesseract.image_to_string(thresh, lang=lang)

    # 화면에 출력
    print("=" * 50)
    print("Tesseract OCR로 추출된 텍스트:")
    print("=" * 50)
    print(extracted_text.strip())
    print("=" * 50)

    return extracted_text


if __name__ == "__main__":
    image_file = "./img2.png"  # 자신의 이미지 경로로 변경
    extract_text_with_tesseract(image_file, lang='kor+eng')  # 한국어+영어 이미지라면 이렇게

