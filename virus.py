import streamlit as st
from io import BytesIO
from datetime import datetime
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image

# Ensure YOLOv5 is in the path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# Import YOLOv5 specific modules

from seg.models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_img_size, cv2,
                           non_max_suppression, scale_coords, strip_optimizer)
from utils.plots import Annotator, colors
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
        weights="seg/runs/train-seg/custom/weights/best.pt",
        source=None,  # opencv image array
        data= 'data/data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.6,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    class_result = []
    conf_result = []

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for im, im0s, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred, out = model(im, augment=augment, visualize=False)
            proto = out[1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Mask plotting
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                im_masks = plot_masks(im[i], masks, mcolors)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)

                # Box plotting
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    class_result.append(int(cls.item()))
                    conf_result.append(conf.item())
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{j + 1} {names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    if update:
        strip_optimizer(weights[0])

    return [im0, class_result, conf_result]

def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

st.set_page_config(
    page_title="App - VSA",
    page_icon="✳"
)

st.title('Aplikasi')
conf_number = st.selectbox('Pilih nilai ambang batas',tuple([0.1,0.25,0.5,0.75,0.95]))
now = datetime.now()
img_files = st.file_uploader(label="Input gambar yang akan diprediksi", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

for n, img_file_buffer in enumerate(img_files):
    if img_file_buffer is not None:
        img_array = create_opencv_image_from_stringio(img_file_buffer)
        open_cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        im0, class_result, conf_result = run(source=open_cv_image, conf_thres=conf_number)

        if class_result is not None:
            res_dict = ['braeburn', 'crimson snow', 'golden', 'golden red', 'granny smith', 'pink lady', 'red', 'red delicious']
            res_text = ''
            if len(class_result) == 0:
                st.error("Maaf tidak ada apel yang ditemukan")
            else:
                res_text += 'Terdapat'
                list_text = []
                for cat_num, cat in enumerate(res_dict):
                    count_cat = class_result.count(cat_num)
                    if count_cat > 0:
                        list_text.append(f' {count_cat} {cat}')
                if len(list_text) > 1:
                    res_text += ','.join(map(str, list_text)) + ' pada gambar yang dimasukkan'
                else:
                    res_text += list_text[0] + ' pada gambar yang dimasukkan'
                st.markdown(f"<p style='text-align: center; font-size:16px'> {res_text} </p>", unsafe_allow_html=True)

            class_apple = [res_dict[i] for i in class_result]
            df_result = pd.DataFrame(list(zip(class_apple, conf_result)), columns=['Kelas', 'Tingkat kepercayaan'])
            df_result.index += 1
            st.table(df_result)

        if im0 is not None:
            st.image(im0, channels="BGR", caption=f'Hasil Segmentasi ({n+1}/{len(img_files)})')

            im1 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            result = Image.fromarray(im1.astype('uint8'), 'RGB')
            buf = BytesIO()
            result.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            col1, col2, col3 = st.columns([3, 3, 3])
            with col2:
                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                st.download_button(
                    label="Unduh gambar",
                    data=byte_im,
                    file_name=f'segmentasi_{dt_string}.png',
                    mime='image/png',
                    use_container_width=True
                )
