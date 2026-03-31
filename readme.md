本工程主要用来实现小说文本转有声小说。

步骤：
1、通过读取文本小说，分析角色，并按照格式输出。
    code：generate_role_by_llm.py
    output：novel_roles.json
2、通过读取文本小说与角色json，分割出旁白与对话，并把对话与角色匹配一起，按照格式输出。
    code：generate_text_by_llm.py
    output：novel_processed.json
3、通过读取对话json，然后生成对应的有声小说音频
    code：generate_audio_by_chattts.py
    output：



