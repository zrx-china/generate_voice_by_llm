import os
import requests
import json
import re
from typing import List, Dict, Any
from tools_call_qianwen import call_qianwen_api_via_requests

# ===================== 配置项 =====================
# 替换为你的通义千问API Key（获取地址：https://dashscope.aliyun.com/）
QWEN_API_KEY = "sk-c9a4649744f246f0877675c62ec3b9f1"
# 小说文本路径（推荐使用绝对路径，例如 "/Users/apple/Dev/Code/novel_10k.txt"）
# NOVEL_TXT_PATH = "/Users/apple/Dev/Code/generate_voice_by_llm/novel_10k.txt"  # mac电脑的环境
NOVEL_TXT_PATH = r"D:\Python\code\generate_voice_by_llm\novel_10k.txt"  # window电脑的环境

# 输出角色档案的JSON路径
OUTPUT_JSON_PATH = "./novel_roles.json"

MODEL_NAME = "qwen-turbo"  # 或 'qwen-plus', 'qwen-max' 等

# ===================== 核心函数 =====================
def read_novel_text(file_path: str, encoding: str = "utf-8") -> str:
    """读取小说TXT文件，清洗多余换行/空格"""
    # 新增：校验文件是否存在
    if not os.path.exists(file_path):
        raise Exception(
            f"文件不存在！请检查路径是否正确：\n当前配置的路径：{file_path}\n"
            "解决方法：\n1. 将小说文件放到该路径下；\n2. 修改代码中 NOVEL_TXT_PATH 为文件的绝对路径（推荐）"
        )
    try:
        with open(file_path, "r", encoding=encoding) as f:
            text = f.read()
            # 清洗文本：合并多余换行、去除无意义空格
            text = re.sub(r"\n+", "\n", text).strip()
            text = re.sub(r"\s+", " ", text)
        return text
    except Exception as e:
        raise Exception(f"读取小说文件失败：{str(e)}")

def split_long_text(text: str, chunk_size: int = 2000) -> List[str]:
    """拆分长文本为若干段（适配大模型输入上限）"""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        # 按句子断句，避免拆分到中间
        if end < text_length:
            # 找最近的句号/换行拆分
            split_pos = text.rfind("。", start, end) + 1
            if split_pos <= start:
                split_pos = text.rfind("\n", start, end) + 1
            if split_pos <= start:
                split_pos = end
            end = split_pos
        chunks.append(text[start:end].strip())
        start = end
    return chunks



def extract_roles_from_chunk(chunk_text: str, api_key: str) -> List[Dict[str, Any]]:
    """从单段文本中提取角色信息"""
    # 核心Prompt：引导大模型输出结构化角色信息（适配ChatTTS）
    prompt = f"""
    请分析以下小说文本，提取所有出场角色的信息，严格按照JSON格式输出（仅输出JSON，无其他解释）：
    要求：
    1. 角色信息包含：
       - name：角色名称（必填，如“顾盼”“陆沉”）；
       - gender：性别（必填，仅填“男/女/未知”）；
       - age：年龄（必填，填具体数字或年龄段，如“28”“30-40”“未知”）；
       - personality：性格特征（必填，如“温柔、冷静、暴躁、内向”）；
       - voice_style：语音风格建议（适配TTS，如“沉稳女声、活力男声、软糯女童、低沉男声”）；
       - description：角色简要描述（可选，100字内）。
    2. 仅提取有具体情节/台词的角色，忽略路人/背景角色；
    3. 若信息未提及，对应字段填“未知”；
    4. JSON格式为数组，示例：
    [
        {{
            "name": "顾盼",
            "gender": "女",
            "age": "26",
            "personality": "温柔、敏感、外冷内热",
            "voice_style": "沉稳女声",
            "description": "女主角，咖啡馆店员，性格细腻，对感情执着"
        }}
    ]

    小说文本：
    {chunk_text}
    """
    # 调用API
    raw_output = call_qianwen_api_via_requests(QWEN_API_KEY, MODEL_NAME, prompt)
    # 清洗输出（去除可能的markdown代码块、多余文字）
    raw_output = raw_output.strip().replace("```json", "").replace("```", "").replace("\\n", "")
    # 解析JSON
    try:
        roles = json.loads(raw_output)
        # 校验格式（过滤无效角色）
        valid_roles = []
        for role in roles:
            if isinstance(role, dict) and "name" in role and role["name"] != "未知":
                # 补全缺失字段
                role.setdefault("gender", "未知")
                role.setdefault("age", "未知")
                role.setdefault("personality", "未知")
                role.setdefault("voice_style", "中性声线")
                role.setdefault("description", "未知")
                valid_roles.append(role)
        return valid_roles
    except json.JSONDecodeError:
        raise Exception(f"大模型输出格式错误，原始输出：{raw_output}")

def merge_roles(role_chunks: List[List[Dict]]) -> List[Dict]:
    """合并多段文本的角色信息（去重，保留最全信息）"""
    role_dict = {}
    for chunk in role_chunks:
        for role in chunk:
            name = role["name"]
            if name not in role_dict:
                role_dict[name] = role
            else:
                # 合并信息：优先保留非“未知”的字段
                for key in ["gender", "age", "personality", "voice_style", "description"]:
                    if role[key] != "未知" and role_dict[name][key] == "未知":
                        role_dict[name][key] = role[key]
    # 转为列表并排序
    merged_roles = sorted(role_dict.values(), key=lambda x: x["name"])
    return merged_roles

def generate_chattts_voice_map(roles: List[Dict]) -> Dict[str, str]:
    """生成ChatTTS角色-音色映射表（适配ChatTTS的voice参数）"""
    voice_map = {
        "中性声线": "neutral",
        "沉稳女声": "female_calm",
        "活力女声": "female_energetic",
        "软糯女童": 2,  # ChatTTS内置音色索引
        "低沉男声": "male_calm",
        "活力男声": "male_energetic",
        "老年男声": 5,
        "老年女声": 6
    }
    role_voice_map = {}
    for role in roles:
        style = role["voice_style"]
        # 匹配最接近的音色
        matched_voice = "neutral"
        for key in voice_map.keys():
            if key in style:
                matched_voice = voice_map[key]
                break
        role_voice_map[role["name"]] = matched_voice
    return role_voice_map

# ===================== 主函数 =====================
def main():
    try:
        # 1. 读取并拆分小说文本
        print("Step 1: 读取小说文本...")
        novel_text = read_novel_text(NOVEL_TXT_PATH)
        print(f"小说总长度：{len(novel_text)} 字符")
        
        print("Step 2: 拆分长文本...")
        text_chunks = split_long_text(novel_text, chunk_size=2000)
        print(f"拆分为 {len(text_chunks)} 段处理")

        # 2. 逐段提取角色信息
        print("Step 3: 调用千问API提取角色信息...")
        all_role_chunks = []
        successful_chunks = 0
        failed_chunks = 0

        for i, chunk in enumerate(text_chunks, 1):
            print(f"  处理第 {i}/{len(text_chunks)} 段...")
            try:
                roles = extract_roles_from_chunk(chunk, QWEN_API_KEY)
                if roles:  # 如果有提取到角色
                    all_role_chunks.append(roles)
                    successful_chunks += 1
                    print(f"    ✅ 成功提取 {len(roles)} 个角色")
                else:
                    all_role_chunks.append([])
                    print(f"    ⚠️  未提取到角色")
                    
            except Exception as e:
                error_msg = str(e)
                
                # 检查是否是内容审核错误
                if "inappropriate content" in error_msg or "For details, see: https://help.aliyun.com/zh/model-studio/error-code#inappropriate-content" in error_msg:
                    print(f"    ⚠️  第{i}段触发内容安全审核，已跳过")
                    all_role_chunks.append([])  # 添加空列表保持索引一致
                    failed_chunks += 1
                    
                    # 可选：记录被跳过的段落信息到日志文件
                    with open("skipped_chunks.log", "a", encoding="utf-8") as log_file:
                        log_file.write(f"=== 跳过的段落 {i} ===\n")
                        log_file.write(f"字符数: {len(chunk)}\n")
                        log_file.write(f"前200字符: {chunk[:200]}...\n")
                        log_file.write(f"错误信息: {error_msg}\n")
                        log_file.write("="*50 + "\n")
                        
                else:
                    # 如果是其他错误，重新抛出
                    print(f"    ❌ 第{i}段处理失败（非内容审核错误）: {error_msg}")
                    raise e  # 重新抛出其他异常

        print(f"\n段落处理完成：成功 {successful_chunks} 段，跳过 {failed_chunks} 段（因内容审核）")

        # 3. 合并角色信息
        print("Step 4: 合并角色信息（去重）...")
        merged_roles = merge_roles(all_role_chunks)
        # 生成ChatTTS音色映射
        voice_map = generate_chattts_voice_map(merged_roles)

        # 4. 保存结果
        result = {
            "roles": merged_roles,
            "chattts_voice_map": voice_map,  # 直接适配ChatTTS的音色映射
            "total_roles": len(merged_roles)
        }
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        # 打印结果
        print("\n===== 角色提取结果 =====")
        print(f"共提取 {len(merged_roles)} 个角色：")
        for role in merged_roles:
            print(f"- {role['name']} | 性别：{role['gender']} | 年龄：{role['age']} | 语音风格：{role['voice_style']}")
        print(f"\n角色档案已保存至：{OUTPUT_JSON_PATH}")
        print(f"\nChatTTS音色映射表：\n{json.dumps(voice_map, ensure_ascii=False, indent=4)}")

    except Exception as e:
        print(f"\n处理失败：{str(e)}")

if __name__ == "__main__":
    main()