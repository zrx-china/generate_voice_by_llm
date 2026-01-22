import requests
import json

def call_qianwen_api(text):
    # 千问API调用参数（示例，需替换为你的实际参数）
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    #url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    headers = {
        "Authorization": "Bearer <sk-c9a4649744f246f0877675c62ec3b9f1>",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-turbo",  # 确认模型名称正确
        "input": {"messages": [{"role": "user", "content": f"提取这段文本的角色信息：{text}"}]},
        "parameters": {"result_format": "message"}
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()  # 抛出HTTP状态码错误（如401/429/500）
        resp_json = response.json()
        
        # 关键：打印完整响应，排查结构
        print("API响应内容：", json.dumps(resp_json, ensure_ascii=False, indent=2))
        
        # 原代码中读取choices的逻辑，先判断字段是否存在
        if "choices" not in resp_json:
            raise ValueError(f"API响应无choices字段，响应内容：{resp_json}")
        
        # 提取角色信息（示例逻辑，需适配你的实际需求）
        result = resp_json["choices"][0]["message"]["content"]
        return result
    
    except requests.exceptions.HTTPError as e:
        raise Exception(f"API HTTP错误：{e}，响应内容：{response.text}")
    except Exception as e:
        raise Exception(f"调用千问API失败：{str(e)}")

# 测试调用（替换为你的文本段）
if __name__ == "__main__":
    test_text = "测试文本，提取角色信息"
    try:
        res = call_qianwen_api(test_text)
        print("角色信息：", res)
    except Exception as e:
        print(f"处理失败：{e}")
