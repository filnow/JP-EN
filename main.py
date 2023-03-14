from paddleocr import PaddleOCR
import gradio as gr

def ocr_jp(img):
    ocr = PaddleOCR(use_angle_cls=True, lang='japan') 
    result = ocr.ocr(img)
    txts = [line[1][0] for line in result[0]]
    output = " ".join(txts)
    return output


demo = gr.Interface(fn=ocr_jp, 
             inputs=gr.inputs.Image(type="numpy"),
             outputs=gr.Textbox(label="OCR Output"),
             examples=[["assets/japan_1.jpg"], ["assets/japan_2.jpg"]],
             )
             
demo.launch()