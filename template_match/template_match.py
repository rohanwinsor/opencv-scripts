import cv2
import numpy as np
import os


def check_if_template_matches(image_path, template_path):
    img = cv2.imread(image_path)
    if img is None:
        assert False, "Image not found"
    alpha = 2.0
    beta = -160
    new = alpha * img + beta
    new = np.clip(new, 0, 255).astype(np.uint8)
    cv2.imwrite("cleaned.png", new)
    command = f"convert {image_path} cleaned.png -compose minus -composite -auto-level -alpha copy watermark1.png"
    os.system(command)
    img = cv2.imread("watermark1.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        print(pt)
        out = cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        if pt != None:
            cv2.imwrite("MATCH.png", out)
            os.system("rm *.png")
            return True
    else:
        os.system("rm *.png")
        return False


if __name__ == "__main__":
    image_path = 'assets/source_image.png'
    template_path = 'assets/template.png'
    out = check_if_template_matches(image_path, template_path)
    print(out)