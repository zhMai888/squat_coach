import cv2
import numpy as np
import random
from ultralytics import YOLO


class SquatDetector:
    def __init__(self, model_path, video_path="yolo11n-pose.pt", threshold=5):
        """
        初始化深蹲检测器

        Args:
            model_path (str): YOLO姿态模型路径
            video_path (str): 视频文件路径
            threshold (int): 关键点y轴变化阈值，用于判定动作阶段切换
        """
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.threshold = threshold

        self.squat_count = 0
        self.looking_for = "down"
        self.lowest_y = None
        self.highest_y = None

        self.cap = cv2.VideoCapture(video_path)
        self.frame_idx = -1
        self.keypoints_buffer = []

        self.squats = []
        self.current_squat = self._init_squat_dict()

    def _init_squat_dict(self):
        """ 初始化当前深蹲数据字典 """
        return {
            "start_idx": None,
            "start": None,
            "down_candidates": [],
            "lowest_idx": None,
            "lowest": None,
            "up_candidates": [],
            "highest_idx": None,
            "highest": None,
        }

    @staticmethod
    def extract_xyc(kp):
        """ 提取关键点的 (x, y, confidence)，如果空则返回0矩阵 """
        if kp is None:
            return np.zeros((18, 3), dtype=np.float32)
        return kp[:, :3].astype(np.float32)

    def relocate_keypoints(self, keypoints):
        """ 重新定位关键点坐标系 """
        if len(keypoints) < 17:
            print("关键点不足，无法重定位")
            return None

        nose = keypoints[0]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]

        # 计算中点和身高
        ankle_mid = (left_ankle[:2] + right_ankle[:2]) / 2
        height = abs(nose[1] - ankle_mid[1]) * 1.08

        # 设置新坐标系：x轴在中点，y轴在头顶，向下为正
        relocated_keypoints = [[0, 0, 1]]  # 头顶
        for kp in keypoints:
            x, y, conf = kp
            x = (x - ankle_mid[0]) / height
            y = (y - (ankle_mid[1] - height)) / height
            relocated_keypoints.append([x, y, conf])

        return relocated_keypoints

    def detect_squats(self):
        """
        执行深蹲检测主流程，返回关键点数组

        Returns:
            np.ndarray: 形状为 (n, 5, 17, 3) 的关键点数据数组
        """
        results = self.model.track(source=self.video_path, show=False, tracker="bytetrack.yaml", persist=True)

        for result in results:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_idx += 1

            # 获取关键点
            if hasattr(result, 'keypoints'):
                keypoints = result.keypoints
                kp = keypoints.data[0].cpu().numpy()
            else:
                kp = None
            self.keypoints_buffer.append(kp)

            if kp is None:
                continue

            # 计算左右臀部中点y坐标作为判断基准
            left_hip = kp[11]
            right_hip = kp[12]
            x1, y1, conf1 = left_hip
            x2, y2, conf2 = right_hip

            if conf1 > 0.5 and conf2 > 0.5:
                mid_y = int((y1 + y2) / 2)

                if self.looking_for == "down":
                    if self.current_squat["start_idx"] is None:
                        self.current_squat["start_idx"] = self.frame_idx
                        self.current_squat["start"] = kp

                    self.current_squat["down_candidates"].append((self.frame_idx, kp))

                    if self.lowest_y is None or mid_y > self.lowest_y:
                        self.lowest_y = mid_y
                    elif self.lowest_y - mid_y > self.threshold:
                        # 下降到最低点后，检测到开始上升
                        self.looking_for = "up"
                        self.highest_y = mid_y

                        prev_idx = self.frame_idx - 1
                        self.current_squat["lowest_idx"] = prev_idx
                        self.current_squat["lowest"] = self.keypoints_buffer[prev_idx]

                elif self.looking_for == "up":
                    self.current_squat["up_candidates"].append((self.frame_idx, kp))

                    if self.highest_y is None or mid_y < self.highest_y:
                        self.highest_y = mid_y
                    elif mid_y - self.highest_y > self.threshold:
                        # 上升结束，完成一次深蹲动作
                        self.squat_count += 1

                        prev_idx = self.frame_idx - 1
                        self.current_squat["highest_idx"] = prev_idx
                        self.current_squat["highest"] = self.keypoints_buffer[prev_idx]

                        # 选取中间帧，限定范围内随机抽取往下和往上的随机帧
                        down_candidates_in_range = [
                            item for item in self.current_squat["down_candidates"]
                            if self.current_squat["start_idx"] <= item[0] <= self.current_squat["lowest_idx"]
                        ]
                        if down_candidates_in_range:
                            down_rand_idx, down_rand_kp = random.choice(down_candidates_in_range)
                        else:
                            down_rand_idx = self.current_squat["start_idx"]
                            down_rand_kp = self.current_squat["start"]

                        up_candidates_in_range = [
                            item for item in self.current_squat["up_candidates"]
                            if self.current_squat["lowest_idx"] <= item[0] <= self.current_squat["highest_idx"]
                        ]
                        if up_candidates_in_range:
                            up_rand_idx, up_rand_kp = random.choice(up_candidates_in_range)
                        else:
                            up_rand_idx = self.current_squat["highest_idx"]
                            up_rand_kp = self.current_squat["highest"]

                        self.current_squat["down_random_idx"] = down_rand_idx
                        self.current_squat["down_random"] = down_rand_kp
                        self.current_squat["up_random_idx"] = up_rand_idx
                        self.current_squat["up_random"] = up_rand_kp

                        self.squats.append(self.current_squat)

                        # 重置状态，准备检测下一次深蹲
                        self.looking_for = "down"
                        self.lowest_y = mid_y
                        self.highest_y = None
                        self.current_squat = self._init_squat_dict()

        # 处理视频结尾时仍在“up”阶段的情况
        if self.looking_for == "up":
            self.squat_count += 1
            self.current_squat["highest_idx"] = self.frame_idx
            self.current_squat["highest"] = self.keypoints_buffer[self.frame_idx]

            down_candidates_in_range = [
                item for item in self.current_squat["down_candidates"]
                if self.current_squat["start_idx"] <= item[0] <= self.current_squat["lowest_idx"]
            ]
            if down_candidates_in_range:
                down_rand_idx, down_rand_kp = random.choice(down_candidates_in_range)
            else:
                down_rand_idx = self.current_squat["start_idx"]
                down_rand_kp = self.current_squat["start"]

            up_candidates_in_range = [
                item for item in self.current_squat["up_candidates"]
                if self.current_squat["lowest_idx"] <= item[0] <= self.current_squat["highest_idx"]
            ]
            if up_candidates_in_range:
                up_rand_idx, up_rand_kp = random.choice(up_candidates_in_range)
            else:
                up_rand_idx = self.current_squat["highest_idx"]
                up_rand_kp = self.current_squat["highest"]

            self.current_squat["down_random_idx"] = down_rand_idx
            self.current_squat["down_random"] = down_rand_kp
            self.current_squat["up_random_idx"] = up_rand_idx
            self.current_squat["up_random"] = up_rand_kp

            self.squats.append(self.current_squat)

        self.cap.release()

        print(f"总共检测到 {self.squat_count} 次深蹲。")

        # 整理成 numpy 数组，形状 (n, 5, 18, 3)
        all_squats_array = []

        for squat in self.squats:
            frames_kp = [
                self.relocate_keypoints(self.extract_xyc(squat["start"])),
                self.relocate_keypoints(self.extract_xyc(squat.get("down_random"))),
                self.relocate_keypoints(self.extract_xyc(squat.get("lowest"))),
                self.relocate_keypoints(self.extract_xyc(squat.get("up_random"))),
                self.relocate_keypoints(self.extract_xyc(squat.get("highest"))),
            ]
            all_squats_array.append(np.array(frames_kp))

        return np.array(all_squats_array, dtype=np.float32)


if __name__ == "__main__":
    model_path = "../yolo11n-pose.pt"
    detector = SquatDetector(model_path)
    squat_keypoints = detector.detect_squats()
    print("关键点数据数组形状:", squat_keypoints.shape)
    print("关键点数据数组内容:", squat_keypoints[0])
