from paddleocr import PaddleOCR
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from paddlespeech.cli.tts.infer import TTSExecutor


def tts(text):
    executor = TTSExecutor()
    audio = executor(text)
    return audio

def ocr_jp(img):
    ocr = PaddleOCR(use_angle_cls=True, lang='japan') 
    result = ocr.ocr(img)
    txts = [line[1][0] for line in result[0]]
    output = " ".join(txts)
    return output

def jp2en(image):
    text = ocr_jp(image)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, output, tts(output)

demo = gr.Interface(fn=jp2en, 
             inputs=gr.inputs.Image(type="numpy"),
             outputs=[gr.Textbox(label="Japanese"), gr.Textbox(label="English"), 'audio'],
             examples=[["assets/japan_1.jpg"], 
                       ["assets/japan_2.jpg"],
                       ["assets/japan_4.jpg"],
                    ]
            )

if __name__ == "__main__":
    demo.launch()