import cv2
from ppadb.client import Client as AdbClient
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'H:\Program Files\Tesseract-OCR\tesseract.exe'


def connect():
    client = AdbClient(host="127.0.0.1", port=5037)  # Default is "127.0.0.1" and 5037
    devices = client.devices()

    if len(devices) == 0:
        print('No devices')
        exit()
    print(f'Connected to {devices[0]}')

    return devices[0]


def screen(device):
    with open('last.png', 'wb') as f:
        f.write(device.screencap())


def get_location(scope, pattern, name=""):
    item = cv2.matchTemplate(scope, pattern, cv2.TM_CCORR_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(item)
    print(f"{name} position best match {max_val} at {max_loc}")
    return max_loc


def drawn_rectangle(scope, location, size=(0, 0)):
    cv2.rectangle(scope, location, (location[0] + size[1], location[1] + size[0]), (0, 0, 255), 2)


def drawn_value_position(scope, location, sample):
    w = location[0] + sample.shape[0]
    h = location[1]
    cv2.rectangle(scope, (w, h), (w + sample.shape[0] * 4, h + sample.shape[1]), (0, 255, 0), 2)


def get_value_box(scope, location, sample):
    w = location[0] + sample.shape[0]
    h = location[1]
    box = scope[h:h + sample.shape[1], w:w + sample.shape[0] * 4]

    # apply mask to white
    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    return cv2.inRange(gray, 210, 255)


def get_int_value(box):
    value = pytesseract.image_to_string(box, config='--oem 1 --psm 6 outputbase digits')
    try:
        value = int(value)
        print(f"output value {value}")
        return value
    except ValueError:
        print(f"output value {value}")
        print("Unable to convert to int")
        return 0


def main(device):
    screen(device)

    last = cv2.imread("last.png", cv2.IMREAD_UNCHANGED)
    src_gold = cv2.imread("src/img/gold_icon.png", cv2.IMREAD_UNCHANGED)
    src_elex = cv2.imread("src/img/elex_icon.png", cv2.IMREAD_UNCHANGED)
    src_dark = cv2.imread("src/img/dark_icon.png", cv2.IMREAD_UNCHANGED)

    print(f"screen size {last.shape}")

    scope = last[0:int(last.shape[0] / 3), 0:int(last.shape[1] / 5)]

    # find gold position
    gold_location = get_location(scope, src_gold, "gold")
    drawn_rectangle(scope, gold_location, src_gold.shape)
    drawn_value_position(scope, gold_location, src_gold)
    gold_box = get_value_box(scope, gold_location, src_gold)
    cv2.imshow("gold", gold_box)

    # find elex position
    elex_location = get_location(scope, src_elex, "elex")
    drawn_rectangle(scope, elex_location, src_elex.shape)
    drawn_value_position(scope, elex_location, src_elex)
    elex_box = get_value_box(scope, elex_location, src_elex)
    cv2.imshow("elex", elex_box)

    # find dark position
    dark_location = get_location(scope, src_dark, "dark")
    drawn_rectangle(scope, dark_location, src_dark.shape)
    drawn_value_position(scope, dark_location, src_dark)
    dark_box = get_value_box(scope, dark_location, src_dark)
    cv2.imshow("dark", dark_box)

    try:
        gold_value = get_int_value(gold_box)
        elex_value = get_int_value(elex_box)
        dark_value = get_int_value(dark_box)

        print("==========================")
        print(f"gold: {gold_value}")
        print(f"elex: {elex_value}")
        print(f"dark: {dark_value}")
        print("==========================")
    except Exception as e:
        print(e)

    cv2.imshow("data", scope)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    device = connect()
    while True:
        main(device)
