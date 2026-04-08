import os
import sys
import numpy as np  # 新增导入

# --- 强制离线 (放在import ChatTTS之前) ---
os.environ['HF_ENDPOINT'] = 'offline'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 切换到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"工作目录：{os.getcwd()}")

# 导入必要的库
import ChatTTS
import torch
import soundfile as sf

# 加载模型
print("正在加载 ChatTTS 模型（强制离线模式）...")
chat = ChatTTS.Chat()

try:
    chat.load(compile=False, source='custom', custom_path=os.path.abspath('./ChatTTS'), device='cpu')
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试文本
test_text = "这是一段测试语音。"
print(f"测试文本：{test_text}")

# 生成语音
try:
    wavs = chat.infer([test_text])
    wav_data = wavs[0]  # 可能是 numpy 数组或 torch tensor

    # 统一转换为 numpy 数组并确保形状正确
    if hasattr(wav_data, 'cpu'):  # torch tensor
        wav_data = wav_data.cpu().numpy()
    # 如果已经是 numpy 数组，直接使用
    if wav_data.ndim == 2:        # 使用 ndim 代替 dim()
        wav_data = wav_data.squeeze(0)
    # 确保类型为 float32
    wav_data = wav_data.astype(np.float32)

    output_wav = "test_output.wav"
    sf.write(output_wav, wav_data, 24000)
    print(f"音频已保存为：{os.path.abspath(output_wav)}")

except Exception as e:
    print(f"推理或保存失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)