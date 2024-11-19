import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# 高斯平滑类
class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianSmoothing, self).__init__()
        self.channels = channels
        # 创建高斯核
        kernel = torch.tensor([
            [1.0, 4.0, 6.0, 4.0, 1.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0]
        ])
        kernel = kernel / kernel.sum()  # 归一化
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # 添加通道维度
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        self.weight = self.weight.half() if x.dtype == torch.float16 else self.weight
        return F.conv2d(x, self.weight, padding=2, groups=self.channels)


# 图片预测类
class ImagePredictor:
    def __init__(self):
        self.model_path = "diffusion.pth"
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型初始化
        self.model = UNet2DModel(
            sample_size=80,  # 宽度
            in_channels=3,
            out_channels=3,
            layers_per_block=2,  # 增加层数
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.model.to(self.device).half()

        # 调度器初始化
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,  # 设置训练时的时间步数
            beta_start=0.00001,  # 更小的起始噪声
            beta_end=0.01,
            beta_schedule='linear'
        )

        # 高斯平滑模块
        self.smoothing = GaussianSmoothing(channels=3, kernel_size=5, sigma=1.0).to(self.device)

        # 预处理步骤
        self.transform = transforms.Compose([
            transforms.Resize((64, 80)),  # 重新调整图像大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # 加载图像并预处理
    def load_image(self, camera_obs):
        # image = Image.open(image_path).convert('RGB')
        # 从 OpenCV (NumPy) 图像转换为 PIL 图像
        pil_image = Image.fromarray(camera_obs)
        return self.transform(pil_image).unsqueeze(0).to(self.device).half()

    # 生成预测图像
    def predict_image(self, camera_obs):
        input_image = self.load_image(camera_obs)

        fixed_timestep = 0.05
        timesteps = torch.full((input_image.shape[0],), fill_value=fixed_timestep, device=self.device).long()

        # 添加噪声
        noise = torch.randn_like(input_image)
        noisy_pic1 = self.scheduler.add_noise(input_image, noise, timesteps)

        # 预测噪声残差
        with torch.no_grad():
            predicted_noise = self.model(noisy_pic1, timesteps).sample

        # 去噪得到预测的下一帧
        predicted_image = noisy_pic1 - predicted_noise

        # 应用高斯平滑
        smoothed_image = self.smoothing(predicted_image)

        # smoothed_image = smoothed_image.to(torch.device("cuda:1"))

        torch.cuda.empty_cache()

        return smoothed_image

    # 显示图像
    def show_image(self, tensor_image, title=None):
        img = tensor_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = (img * 0.5) + 0.5  # 反标准化
        plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()


# 使用示例
if __name__ == "__main__":
    predictor = ImagePredictor()

    # 预测并显示图像
    predicted_image = predictor.predict_image('./292.jpg')
    predictor.show_image(predicted_image, title="Predicted and Smoothed Image")
