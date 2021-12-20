import cv2
from ngraph import ctc_greedy_decoder
from numpy import fromstring, where
from openvino.inference_engine import IECore
from imutils import paths, resize

TEST_PATH = "houses"
CONF = 0.4
MODEL_FRAME_SIZE = 704

pColor = (0, 0, 255)
rectThinkness = 1

alpha_numeric_symbols = "#1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"


address_model_xml = "./horizontal-text-detection-0001.xml"
address_model_bin = "./horizontal-text-detection-0001.bin"

tr_model_xml = "./text-recognition-0014.xml"
tr_model_bin = "./text-recognition-0014.bin"


device = "CPU"


def drawText(frame, scale, rectX, rectY, rectColor, text):

    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)

    top = max(rectY - rectThinkness, textSize[0])

    cv2.putText(
        frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3
    )


def ctc_greedy_decoder(results):
    content = ""
    for element in results:
        _max = max(element[0])
        _max_index = int(where(element[0] == _max)[0])
        if _max_index != 0:
            content += alpha_numeric_symbols[_max_index]
    return content


def addressDetection(
    frame, address_execution_net, address_input_blob, tr_execution_net, tr_input_blob
):

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    address_blob = cv2.dnn.blobFromImage(
        frame, size=(MODEL_FRAME_SIZE, MODEL_FRAME_SIZE), ddepth=cv2.CV_8U
    )
    address_results = address_execution_net.infer(
        inputs={address_input_blob: address_blob}
    ).get("boxes")

    if address_results.any():
        for detection in address_results:
            if detection[0] == 0:
                break
            conf = detection[4]
            if conf < CONF:
                continue
            xmin = int(detection[0] * frame_width / MODEL_FRAME_SIZE)
            ymin = int(detection[1] * frame_height / MODEL_FRAME_SIZE)
            xmax = int(detection[2] * frame_width / MODEL_FRAME_SIZE)
            ymax = int(detection[3] * frame_height / MODEL_FRAME_SIZE)
            xmin = max(0, xmin - 5)
            ymin = max(0, ymin - 5)
            xmax = min(xmax + 5, frame_width - 1)
            ymax = min(ymax + 5, frame_height - 1)
            trImg = frame[ymin : ymax + 1, xmin : xmax + 1]
            gray_image = cv2.cvtColor(trImg, cv2.COLOR_BGR2GRAY)
            blob = cv2.dnn.blobFromImage(gray_image, size=(128, 32), ddepth=cv2.CV_8U)
            tr_results = tr_execution_net.infer(inputs={tr_input_blob: blob}).get(
                "logits"
            )
            rectW = xmax - xmin
            content = ctc_greedy_decoder(tr_results)
            print("CONTENT: ", content)
            drawText(frame, rectW * 0.008, xmin, ymin, pColor, content)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), pColor, rectThinkness)

    showImg = resize(frame, height=750)
    cv2.imshow("showImg", showImg)


def main():

    ie = IECore()

    address_neural_net = ie.read_network(
        model=address_model_xml, weights=address_model_bin
    )
    if address_neural_net is not None:
        address_input_blob = next(iter(address_neural_net.input_info))
        address_neural_net.batch_size = 1
        address_execution_net = ie.load_network(
            network=address_neural_net, device_name=device.upper()
        )

    print(address_neural_net.input_info[address_input_blob].input_data.shape)

    tr_neural_net = ie.read_network(model=tr_model_xml, weights=tr_model_bin)
    if tr_neural_net is not None:
        tr_input_blob = next(iter(tr_neural_net.input_info))
        tr_neural_net.batch_size = 1
        tr_execution_net = ie.load_network(
            network=tr_neural_net, device_name=device.upper()
        )

    for imagePath in paths.list_images(TEST_PATH):
        print(imagePath)
        img = cv2.imread(imagePath)
        if img is None:
            continue

        addressDetection(
            img,
            address_execution_net,
            address_input_blob,
            tr_execution_net,
            tr_input_blob,
        )
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
