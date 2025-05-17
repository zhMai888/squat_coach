from dataclasses import dataclass
from gtts import gTTS
from pose import SquatDetector
from predict_standard import Standard
from chat import load, Prompt


@dataclass
class ModelParm():
    video_path: str
    model_path: str = "../yolo11n-pose.pt"
    threshold: int = 5


class Action():
    def __init__(self):
        self.all_messages = []

    def run(self,config: ModelParm):
        squat_detector = SquatDetector(config.video_path, config.model_path, config.threshold)
        squat_keypoints = squat_detector.detect_squats()
        for i in range(squat_keypoints.shape[0]):
            messages = []
            for j in range(5):
                photo_keypoint = squat_keypoints[i][j]
                standard = Standard(photo_keypoint)     #（5，17，3）
                res = standard.evaluate_squat()         # （5，8，4）
                messages.append(res)
            self.all_messages.append(messages)

        # 生成提示
        prompt = Prompt()
        data_action = prompt.load_prompt()
        tokenizer, model = load()
        history = None
        for i in range(len(data_action.keys())):
            if i == 0:
                p = prompt.generate_prompt(data_action["fewshot" + str(i + 1)], first_time=True)
            else:
                p = prompt.generate_prompt(data_action["fewshot" + str(i + 1)])

            response, history = model.chat(tokenizer, p, history=history)

        print("准备完成，开始检测")

        # 真正开始预测
        true_data = prompt.predict_prompt(self.all_messages)
        response, history = model.chat(tokenizer, true_data, history=history)

        print(response)

        #开始转换成语言
        print("生成完成，正在转为语言输出")
        tts = gTTS(text=response, lang='zh-cn')
        tts.save("output.mp3")
        print("音频已保存为 output.mp3")



if __name__ == "__main__":
    a = ModelParm(video_path="images/test_video2.mp4")
    action = Action()
    action.run(a)