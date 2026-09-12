# 部署 V8.7
1. 解压 ZIP。
2. 把里面全部文件直接上传 GitHub 仓库根目录；不需要自己建文件夹。
3. Railway Redeploy。
4. `/api/health` 必须显示 8.7 / server-self-healing。
5. 配置 DeepSeek：Railway Variables 添加 DEEPSEEK_API_KEY 和 DEEPSEEK_MODEL。
6. `/api/deepseek/test` 检查 AI 是否连通。
