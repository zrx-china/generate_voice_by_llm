import requests
import json
from http import HTTPStatus  # 用于状态码判断

def call_qianwen_api_via_requests(api_key: str, model: str, prompt: str) -> str:
    """
    使用requests库直接调用通义千问API（推荐用于简单请求）
    
    标准请求方式参考官方文档：https://help.aliyun.com/document_detail/2712581.html[citation:2]
    
    Args:
        api_key: 你的DashScope API Key (sk-开头)
        model: 模型名称，如 'qwen-turbo', 'qwen-plus', 'qwen-max'
        prompt: 用户输入的文本提示
    
    Returns:
        API返回的文本内容
    """
    # 标准API端点 (与你的generate_*.py文件一致)
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    # 标准请求头 - 注意API密钥格式（不要尖括号！）
    headers = {
        "Authorization": f"Bearer {api_key}",  # 关键修正：移除了尖括号
        "Content-Type": "application/json"
    }
    
    # 标准请求体 (通过messages调用，这是最新推荐方式)[citation:2]
    payload = {
        "model": model,
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "result_format": "text",  # 或 "message" 格式更完整[citation:10]
            "temperature": 0.1  # 低温度保证输出稳定
        }
    }
    
    try:
        # 发送请求，设置合理超时
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # 检查HTTP状态码
        
        # 解析响应
        resp_json = response.json()
        
        # 打印完整响应结构以便调试
        print("完整的API响应结构：")
        print(json.dumps(resp_json, ensure_ascii=False, indent=2))
        
        # 关键：根据不同的result_format提取结果
        if resp_json.get("output"):
            output = resp_json["output"]
            
            # 方式1: result_format为"text"时的解析 (你的代码原有方式)
            if "text" in output and output["text"]:
                return output["text"]
            
            # 方式2: result_format为"message"时的解析 (推荐格式)[citation:10]
            if "choices" in output and output["choices"]:
                message = output["choices"][0].get("message", {})
                if message and "content" in message:
                    return message["content"]
        
        # 如果以上方式都没提取到，抛出异常
        raise ValueError(f"无法从API响应中提取内容，响应结构异常：{resp_json}")
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP错误 ({response.status_code})"
        if response.text:
            try:
                error_detail = json.loads(response.text)
                error_msg += f": {error_detail.get('message', response.text)}"
            except:
                error_msg += f": {response.text}"
        raise Exception(f"API调用失败 - {error_msg}")
    except json.JSONDecodeError as e:
        raise Exception(f"API返回的不是有效JSON: {response.text[:200]}")
    except Exception as e:
        raise Exception(f"调用千问API失败：{str(e)}")

# 另一种选择：使用官方SDK的调用方式（更简洁，但需额外安装）
# def call_qianwen_api_via_sdk(api_key: str, prompt: str):
#     """
#     使用阿里云官方DashScope SDK调用（需要安装：pip install dashscope）
#     这是官方推荐方式，代码更简洁[citation:2]
#     """
#     try:
#         import dashscope
        
#         # 设置API密钥
#         dashscope.api_key = api_key
        
#         # 直接调用（通过prompt方式，简单直接）
#         response = dashscope.Generation.call(
#             model='qwen-turbo',
#             prompt=prompt
#         )
        
#         if response.status_code == HTTPStatus.OK:
#             return response.output.text
#         else:
#             return f"Error: {response.code} - {response.message}"
            
#     except ImportError:
#         return "错误：请先安装dashscope SDK (pip install dashscope)"
#     except Exception as e:
#         return f"SDK调用失败：{str(e)}"

# 测试调用（使用requests方式）
if __name__ == "__main__":
    # 配置参数 - 替换为你的实际信息
    MY_API_KEY = "sk-c9a4649744f246f0877675c62ec3b9f1"  # 注意：不要尖括号！
    MODEL_NAME = "qwen-turbo"  # 或 'qwen-plus', 'qwen-max' 等
    
    # 测试文本 - 可以使用你的小说角色提取提示
    test_prompt = "请分析以下文本中的角色信息：顾盼是一个温柔的女孩，陆沉是个冷静的商人。他们在一家咖啡馆相遇。"
    
    print("正在测试标准千问API调用...")
    print(f"模型：{MODEL_NAME}")
    print(f"提示词：{test_prompt[:50]}...")
    print("-" * 50)
    
    try:
        # 方式1: 使用requests直接调用（推荐）
        result = call_qianwen_api_via_requests(MY_API_KEY, MODEL_NAME, test_prompt)
        print("✅ API调用成功！")
        print("提取的角色信息：")
        print(result)
        
        print("\n" + "=" * 50)
        print("可选：使用官方SDK方式调用...")
        
        # # 方式2: 使用官方SDK调用（如果已安装）
        # sdk_result = call_qianwen_api_via_sdk(MY_API_KEY, test_prompt)
        # print(f"SDK调用结果：{sdk_result}")
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        print("\n故障排除建议：")
        print("1. 检查API Key是否正确且未过期")
        print("2. 确认API Key格式为 'sk-xxxxxxxx'，不要加尖括号")
        print("3. 检查网络连接，确保能访问 https://dashscope.aliyuncs.com")
        print("4. 查看阿里云控制台，确认服务已开通且有余量")