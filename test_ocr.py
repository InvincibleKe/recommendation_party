import pytesseract
from PIL import Image
if __name__ == '__main__':
    image = Image.open('Data/test6.jpeg')
    code = pytesseract.image_to_string(image, lang='chi_sim')
    print(code)