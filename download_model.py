from huggingface_hub import snapshot_download

# 模型ID (注意大小写)
model_id = "2noise/ChatTTS"

# 下载到哪个目录，这里的路径可以按需修改
local_dir = "./asset"

# 开始下载
snapshot_download(repo_id=model_id, local_dir=local_dir)