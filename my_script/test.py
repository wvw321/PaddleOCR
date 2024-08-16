from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
# ocr = PaddleOCR(use_angle_cls=True, lang='ru',use_gpu=True)


ocr = PaddleOCR(lang="ru",use_angle_cls=True)  # need to run only once to download and load model into memory
# img_path = 'dataset/SimpleDataSet/valid_data/images/s-00012-0036.jpg'
# result = ocr.ocr(img_path, det=False)
# print(result)


# result = ocr.ocr(img_path, cls=True)
# for idx in range(len(result)):
#     res = result[idx]
#     for line in res:
#         print(line)
img_path = "F:/projects/git/doctr/pass/3.jpg"
result = ocr.ocr(img_path, rec=False)
image = Image.open(img_path).convert('RGB')
im_show = draw_ocr(image, boxes=result)
im_show = Image.fromarray(im_show)
im_show.show()
print()
# directory = "F:/projects/git/tesseract/снилс для валидации"
# files = os.listdir(directory)
# data = {}
# for x in files:
#     result = ocr.ocr(directory + "/" + x, det=False)[0][0][0].lower().rstrip()
#     data[x] = result
#
# with open("labels_val.txt", "w", encoding="utf-8") as file:
#     for x in data.keys():
#         file.write(x + "\t" + data[x] + "\n")
#
#
# with open(directory + "\labels_val.txt", "w", encoding="utf-8") as file:
#     for x in data.keys():
#         file.write(x + "\t" + data[x] + "\n")
