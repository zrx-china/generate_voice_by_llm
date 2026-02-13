import json
import os
import torch
import ChatTTS
from pydub import AudioSegment
from typing import List, Dict

# 初始化ChatTTS模型
chat = ChatTTS.Chat()
chat.load_models()  # 自动下载并加载模型，首次运行需联网


def get_chattts_speaker_params(emotion: str, speed: float) -> Dict:
    """
    根据情感和语速生成ChatTTS对应的参数
    :param emotion: 情感标签（neutral/happy/sad/angry/calm/surprised）
    :param speed: 语速（0.8~1.2）
    :return: ChatTTS参数字典
    """
    # 情感对应的风格参数（可根据实际效果调整）
    emotion_map = {
        "neutral": {"temperature": 0.3, "top_P": 0.7, "top_K": 20},
        "happy": {"temperature": 0.7, "top_P": 0.8, "top_K": 15},
        "sad": {"temperature": 0.2, "top_P": 0.6, "top_K": 25},
        "angry": {"temperature": 0.8, "top_P": 0.9, "top_K": 10},
        "calm": {"temperature": 0.1, "top_P": 0.5, "top_K": 30},
        "surprised": {"temperature": 0.9, "top_P": 0.85, "top_K": 12}
    }
    
    # 语速映射（ChatTTS的speed范围0.5~2.0，这里做线性转换）
    chat_speed = speed  # 直接复用配置的语速，也可调整：如 speed * 1.2
    
    # 生成speaker_id（固定一个音色，也可根据角色切换）
    speaker_id = torch.randint(0, 10000, (1,)).item()
    
    params = {
        "text": "",
        "skip_refine_text": False,
        "params_infer_code": {
            "spk_id": speaker_id,
            "temperature": emotion_map[emotion]["temperature"],
            "top_P": emotion_map[emotion]["top_P"],
            "top_K": emotion_map[emotion]["top_K"],
            "speed": chat_speed
        },
        "params_refine_text": {
            "prompt": f"[{emotion}]"  # 给文本添加情感提示
        }
    }
    return params


def generate_voice_from_json(json_path: str, output_path: str = "novel_voice.wav"):
    """
    从novel_processed.json生成语音并合并为完整音频
    :param json_path: novel_processed.json文件路径
    :param output_path: 最终合并后的音频文件路径
    """
    # 1. 读取JSON文件
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON文件不存在：{json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        novel_data: List[Dict] = json.load(f)
    
    # 2. 临时音频片段存储列表
    temp_audio_segments = []
    temp_dir = "temp_audio_segments"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 3. 逐段生成语音
    for idx, segment in enumerate(novel_data):
        try:
            text = segment["text"].strip()
            speaker = segment["speaker"]
            emotion = segment["emotion"]
            speed = segment["speed"]
            
            if not text:  # 跳过空文本
                continue
            
            print(f"正在生成 [{idx+1}/{len(novel_data)}] - 说话人：{speaker} - 情感：{emotion}")
            
            # 获取ChatTTS参数
            tts_params = get_chattts_speaker_params(emotion, speed)
            tts_params["text"] = text
            
            # 生成语音（返回音频数据，采样率24000）
            wav = chat.infer(
                [text],
                skip_refine_text=tts_params["skip_refine_text"],
                params_infer_code=tts_params["params_infer_code"],
                params_refine_text=tts_params["params_refine_text"]
            )
            
            # 将tensor转为音频片段并保存临时文件
            wav_tensor = wav[0].cpu().numpy()
            temp_file = os.path.join(temp_dir, f"segment_{idx}.wav")
            
            # 保存为WAV文件（使用pydub）
            audio_segment = AudioSegment(
                wav_tensor.tobytes(),
                frame_rate=24000,
                sample_width=wav_tensor.dtype.itemsize,
                channels=1
            )
            audio_segment.export(temp_file, format="wav")
            temp_audio_segments.append(audio_segment)
            
        except Exception as e:
            print(f"生成第{idx+1}段语音失败：{str(e)}")
            continue
    
    # 4. 合并所有音频片段
    if not temp_audio_segments:
        raise ValueError("未生成任何音频片段")
    
    merged_audio = AudioSegment.empty()
    for seg in temp_audio_segments:
        merged_audio += seg
    
    # 5. 保存最终音频文件
    merged_audio.export(output_path, format="wav")
    print(f"\n音频生成完成！文件保存至：{os.path.abspath(output_path)}")
    
    # 6. 清理临时文件（可选）
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)


if __name__ == "__main__":
    # 配置文件路径
    JSON_FILE_PATH = "novel_processed.json"  # 你的JSON文件路径
    OUTPUT_AUDIO_PATH = "novel_full_voice.wav"  # 输出音频路径
    
    try:
        # 生成语音
        generate_voice_from_json(JSON_FILE_PATH, OUTPUT_AUDIO_PATH)
    except Exception as e:
        print(f"程序执行失败：{str(e)}")