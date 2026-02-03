import requests
import json
import os
from typing import List, Dict

from tools_call_qianwen import call_qianwen_api_via_requests



MODEL_NAME = "qwen-turbo"  # 或 'qwen-plus', 'qwen-max' 等


def preprocess_novel_text(raw_text: str, api_key: str,novel_roles_path: str) -> List[Dict]:
    """
    调用大模型预处理小说文本，返回结构化的角色/情感/语速标注数据
    
    Args:
        raw_text: 原始小说文本
        api_key: 通义千问API Key（需自行申请：https://dashscope.aliyun.com/）
    
    Returns:
        结构化列表，每个元素包含text/speaker/emotion/speed
    """
    with open(novel_roles_path, "r", encoding="utf-8") as f:
        role_data = json.load(f)
    # role_prompt = role_data["roles"]

    # 1. 构造大模型提示词
    prompt = f"""
    {role_data}
    请严格按照以下要求处理小说文本，仅输出JSON格式结果（不要额外解释）：
    1. 文本清洗：去除无关空格/重复标点，保留完整语义；
    2. 分段断句：按自然语义拆分，每段不超过200字；
    3. 角色标注：区分「旁白」和具体角色名（如“顾盼”“陆沉”）；
    4. 情感标注：仅用 neutral/happy/sad/angry/calm/surprised 标注；
    5. 语速建议：0.8~1.2之间的浮点数（默认1.0）。

    小说文本：
    {raw_text}

    输出格式示例：
    [
        {{
            "text": "他缓缓抬起头，眼中满是悲伤。",
            "speaker": "旁白",
            "emotion": "sad",
            "speed": 0.9
        }},
        {{
            "text": "你为什么要离开我？",
            "speaker": "男主",
            "emotion": "angry",
            "speed": 1.1
        }}
    ]
    """
    # 2. 调用通义千问API
    raw_output = call_qianwen_api_via_requests(api_key, MODEL_NAME, prompt)
    # 清洗输出（去除可能的markdown代码块、多余文字）
    raw_output = raw_output.strip().replace("```json", "").replace("```", "").replace("\\n", "")
    # return raw_output

    # ========== 关键修复：把JSON字符串解析成字典列表 ==========
    try:
        processed_data = json.loads(raw_output)  # 解析为列表/字典
        # 校验解析结果是否为列表（确保格式符合要求）
        if not isinstance(processed_data, list):
            raise ValueError(f"大模型输出不是列表格式，原始输出：{raw_output}")
        # 校验每个元素是否为字典，且包含必要字段
        required_fields = ["text", "speaker", "emotion", "speed"]
        for item in processed_data:
            if not isinstance(item, dict):
                raise ValueError(f"片段不是字典：{item}")
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"缺失必要字段{field}：{item}")
        return processed_data  # 返回解析后的字典列表
    except json.JSONDecodeError as e:
        raise Exception(f"大模型输出不是合法JSON，原始输出：{raw_output}，错误：{e}")
    # # 2. 调用通义千问API
    # url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    # headers = {
    #     "Authorization": f"Bearer {api_key}",
    #     "Content-Type": "application/json"
    # }
    # payload = {
    #     "model": "qwen-turbo",  # 轻量版，速度快、成本低
    #     "input": {
    #         "messages": [
    #             {"role": "user", "content": prompt}
    #         ]
    #     },
    #     "parameters": {
    #         "result_format": "text",
    #         "temperature": 0.1,  # 降低随机性，保证输出格式稳定
    #         "top_p": 0.9
    #     }
    # }

    # try:
    #     # 发送请求
    #     response = requests.post(url, headers=headers, json=payload, timeout=30)
    #     response.raise_for_status()  # 抛出HTTP错误
    #     result = response.json()

    #     # 关键修改：用json.dumps格式化输出，indent=4表示缩进4个空格，ensure_ascii=False显示中文
    #     formatted_payload = json.dumps(
    #         payload,
    #         indent=4,        # 缩进4个空格，层级清晰
    #         ensure_ascii=False,  # 保留中文，不转义
    #         sort_keys=False   # 不打乱原有字段顺序
    #     )

    #     print(f"--1--原始文本：\n{formatted_payload}")
        
    #     # 关键步骤：解析output.text中的JSON字符串为Python列表
    #     if "output" in result and "text" in result["output"]:
    #         # 把带转义符的字符串转成真正的列表
    #         text_data = json.loads(result["output"]["text"])
    #         # 替换原字段，用解析后的列表替代字符串
    #         result["output"]["text"] = text_data
    #     # 关键修改：用json.dumps格式化输出，indent=4表示缩进4个空格，ensure_ascii=False显示中文
    #     formatted_result = json.dumps(
    #         result,
    #         indent=4,        # 缩进4个空格，层级清晰
    #         ensure_ascii=False,  # 保留中文，不转义
    #         sort_keys=False   # 不打乱原有字段顺序
    #     )
    #     print(f"--2--阿里大模型反馈的结果：\n{formatted_result}")
    #     # 3. 解析返回结果
    #     raw_output = result["output"]["text"]
    #     print(f"--3--变量类型：{type(raw_output)}")

    #     return raw_output

    # except requests.exceptions.RequestException as e:
    #     raise Exception(f"API调用失败：{str(e)}")
    # except json.JSONDecodeError as e:
    #     raise Exception(f"大模型输出格式错误，原始输出：{raw_output}，错误：{str(e)}")
    # except KeyError as e:
    #     raise Exception(f"API返回字段缺失：{str(e)}，原始返回：{result}")

def read_novel_from_txt(file_path: str, encoding: str = "utf-8") -> str:
    """
    从TXT文件读取小说文本
    
    Args:
        file_path: TXT文件路径（绝对路径/相对路径）
        encoding: 文件编码，默认utf-8，若乱码可尝试gbk/gb2312
    
    Returns:
        读取的文本内容
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    # 检查文件是否是TXT
    if not file_path.endswith(".txt"):
        raise ValueError("仅支持读取.txt格式文件")
    
    # 读取文件
    try:
        with open(file_path, "r", encoding=encoding) as f:
            # 读取所有内容并清洗多余换行/空格
            text = f.read().replace("\n\n", "\n").strip()
        return text
    except UnicodeDecodeError:
        raise Exception(f"文件编码错误，尝试用gbk编码读取（当前编码：{encoding}）")
    except Exception as e:
        raise Exception(f"读取文件失败：{str(e)}")

# ===================== 调用示例 =====================
if __name__ == "__main__":
    # 1. 配置参数
    MY_API_KEY = "sk-c9a4649744f246f0877675c62ec3b9f1"  # 替换为你的通义千问API Key
    NOVEL_TXT_PATH = "/Users/apple/Dev/Code/generate_voice_by_llm/novel_sample.txt"  # mac电脑的环境
    NOVEL_ROLES_PATH = "/Users/apple/Dev/Code/generate_voice_by_llm/novel_roles.json"  # mac电脑的环境
    NOVEL_PROCESSED_PATH= "/Users/apple/Dev/Code/generate_voice_by_llm/novel_processed.json" # mac电脑的环境
    
    try:
        # 2. 从TXT文件读取小说文本
        print(f"正在读取文件：{NOVEL_TXT_PATH}")
        novel_raw_text = read_novel_from_txt(NOVEL_TXT_PATH, encoding="utf-8")
        print(f"文件读取完成，文本长度：{len(novel_raw_text)} 字符")

        # 3. （可选）长文本拆分：通义千问turbo单轮最大支持8k字符，超过则按章节拆分
        max_text_length = 2000
        if len(novel_raw_text) > max_text_length:
            print("检测到长文本，自动拆分处理...")
            # 按章节/换行拆分（简单拆分逻辑，可根据小说格式优化）
            text_chunks = []
            current_chunk = ""
            for line in novel_raw_text.split("\n"):
                if len(current_chunk + line) < max_text_length:
                    current_chunk += line + "\n"
                else:
                    text_chunks.append(current_chunk.strip())
                    current_chunk = line + "\n"
            if current_chunk:
                text_chunks.append(current_chunk.strip())
        else:
            text_chunks = [novel_raw_text]

        # 4. 批量预处理每个文本块
        all_processed_segments = []
        for i, chunk in enumerate(text_chunks, 1):
            print(f"\n正在预处理第{i}个文本块...")
            processed_chunk = preprocess_novel_text(chunk, MY_API_KEY,NOVEL_ROLES_PATH)
            all_processed_segments.extend(processed_chunk)

        # 5. 打印预处理结果
        print("\n===== 小说文本预处理结果 =====")
        for i, seg in enumerate(all_processed_segments, 1):
            print(f"\n【片段{i}】")
            print(f"文本：{seg['text']}")
            print(f"说话人：{seg['speaker']}")
            print(f"情感：{seg['emotion']}")
            print(f"语速：{seg['speed']}")

        # （可选）将预处理结果保存为JSON文件
        with open(NOVEL_PROCESSED_PATH, "w", encoding="utf-8") as f:
            json.dump(all_processed_segments, f, ensure_ascii=False, indent=4)
        print(f"\n预处理结果已保存至：{NOVEL_PROCESSED_PATH}")

    except Exception as e:
        print(f"处理失败：{str(e)}")