import math
import numpy as np

class Standard:
    def __init__(self, model: np.ndarray):
        assert model.shape == (17, 3)
        self.model = model

    # 计算中点
    def midpoint(self, p1, p2):
        point = (p1[:2] + p2[:2]) / 2
        confidence = (p1[2] + p2[2]) / 2  # 平均置信度
        return np.array([point[0], point[1], confidence])

    # 计算两点之间的距离
    def distance(self, p1, p2):
        return np.linalg.norm(p1[:2] - p2[:2])

    # 计算三点之间的角度
    def angle_between(self, p1, p2, p3):
        a = self.distance(p2, p3)
        b = self.distance(p1, p3)
        c = self.distance(p1, p2)
        if a * b == 0:
            return 0
        return math.degrees(math.acos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))

    # 计算标准
    def evaluate_squat(self):
        # 关键点
        left_shoulder = self.model[5]
        right_shoulder = self.model[6]
        left_hip = self.model[11]
        right_hip = self.model[12]
        left_knee = self.model[13]
        right_knee = self.model[14]
        left_ankle = self.model[15]
        right_ankle = self.model[16]

        # 中点坐标计算
        shoulder_mid = self.midpoint(left_shoulder, right_shoulder)
        hip_mid = self.midpoint(left_hip, right_hip)
        knee_mid = self.midpoint(left_knee, right_knee)
        ankle_mid = self.midpoint(left_ankle, right_ankle)
        toe_mid = self.midpoint(
            np.array([left_ankle[0], left_ankle[1] - 0.01, left_ankle[2]]),
            np.array([right_ankle[0], right_ankle[1] - 0.01, right_ankle[2]])
        )

        # 躯干角度（肩膀到髋关节）
        trunk_angle = min(abs(math.degrees(math.atan2(hip_mid[1] - shoulder_mid[1],hip_mid[0] - shoulder_mid[0]))),
                          math.degrees(math.atan2(-(shoulder_mid[1] - hip_mid[1]),shoulder_mid[0] - hip_mid[0])))

        trunk_confidence = (shoulder_mid[2] + hip_mid[2]) / 2

        # 髋角度（髋关节-膝盖-脚踝）
        hip_angle = self.angle_between(hip_mid, knee_mid, ankle_mid)
        hip_confidence = (hip_mid[2] + knee_mid[2] + ankle_mid[2]) / 3

        # 膝盖-脚尖对齐（是否稳定）
        knee_to_toe_distance = abs(knee_mid[0] - toe_mid[0])
        knee_to_toe_confidence = (knee_mid[2] + toe_mid[2]) / 2

        # 脚跟稳定性
        heel_stability = abs(left_ankle[1] - right_ankle[1])
        heel_confidence = (left_ankle[2] + right_ankle[2]) / 2

        # 深蹲深度
        squat_depth = abs(hip_mid[1] - knee_mid[1])
        squat_depth_confidence = (hip_mid[2] + knee_mid[2]) / 2

        # 肩宽和膝宽
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        knee_width = abs(left_knee[0] - right_knee[0])
        shoulder_knee_confidence = (left_shoulder[2] + right_shoulder[2] + left_knee[2] + right_knee[2]) / 4

        """
        (信息, 置信度, 评估结果, 偏置量)
        trunk_angle: 躯干角度
        hip_angle: 髋角度
        knee_to_toe_distance: 膝盖到脚尖的距离
        heel_stability: 脚跟稳定性
        squat_depth: 蹲起深度
        shoulder_symmetry: 肩部对称性
        knee_symmetry: 膝盖对称性
        knee_width: 膝盖宽度
        """

        results = [
            [trunk_angle, trunk_confidence, 1, 0] if 60 <= trunk_angle <= 85 else [trunk_angle, trunk_confidence, 0, (trunk_angle - 60) * (trunk_angle < 60) + (trunk_angle - 85) * (trunk_angle > 85)],
            [hip_angle, hip_confidence, 1, 0] if 30 <= hip_angle <= 75 else [hip_angle, hip_confidence, 0, (hip_angle - 30) * (hip_angle < 30) + (hip_angle - 75) * (hip_angle > 75)],
            [knee_to_toe_distance, knee_to_toe_confidence, 1, 0] if knee_to_toe_distance < 0.07 else [knee_to_toe_distance, knee_to_toe_confidence, 0, (knee_to_toe_distance - 0.07)],
            [heel_stability, heel_confidence, 1, 0] if abs(heel_stability) < 0.04 else [heel_stability, heel_confidence, 0, (heel_stability - 0.04)],
            [squat_depth, squat_depth_confidence, 1, 0] if 0.05 <= squat_depth <= 0.15 else [squat_depth, squat_depth_confidence, 0, (squat_depth - 0.05) * (squat_depth < 0.05) + (squat_depth - 0.15) * (squat_depth > 0.15)],
            [abs(left_shoulder[1] - right_shoulder[1]), (left_shoulder[2] + right_shoulder[2]) / 2, 1, 0] if abs(left_shoulder[1] - right_shoulder[1]) < 0.02 else [abs(left_shoulder[1] - right_shoulder[1]),(left_shoulder[2] + right_shoulder[2]) / 2, 0,abs(left_shoulder[1] - right_shoulder[1]) - 0.02],
            [abs(left_knee[1] - right_knee[1]), (left_knee[2] + right_knee[2]) / 2, 1, 0] if abs(left_knee[1] - right_knee[1]) < 0.02 else [abs(left_knee[1] - right_knee[1]),(left_knee[2] + right_knee[2]) / 2, 0,abs(left_knee[1] - right_knee[1]) - 0.02],
            [knee_width, shoulder_knee_confidence, 1, 0] if abs(knee_width - shoulder_width) < 0.05 else [knee_width,shoulder_knee_confidence,0, abs(knee_width - shoulder_width) - 0.05]
        ]

        np.set_printoptions(suppress=True, precision=3)

        results = np.round(results,3).tolist()

        return results

if __name__ == '__main__':
    model = np.array([
        [-0.022, 0.074, 0.996], [-0.002, 0.054, 0.986], [-0.042, 0.052, 0.987],
        [0.026, 0.076, 0.847], [-0.071, 0.073, 0.808], [0.054, 0.208, 0.994], [-0.090, 0.208, 0.997],
        [0.087, 0.361, 0.959], [-0.131, 0.379, 0.979], [0.101, 0.484, 0.947], [-0.131, 0.517, 0.967],
        [0.040, 0.493, 0.998], [-0.066, 0.494, 0.998], [0.074, 0.747, 0.995], [-0.104, 0.750, 0.996],
        [0.127, 0.998, 0.955], [-0.127, 1.002, 0.963]
    ])

    standard_model = Standard(model)
    res = standard_model.evaluate_squat()
    print("评估结果：")
    print(res)
