@echo off
chcp 65001 >nul
echo ========================================
echo 鼹鼠分类智能体 - 一键启动 (ai环境)
echo ========================================
echo.

REM 激活 ai 环境
echo [1/4] 激活 ai conda 环境...
call conda activate ai
if errorlevel 1 (
    echo 警告：无法激活 ai 环境，使用当前环境
)

REM 检查依赖
echo.
echo [2/4] 检查依赖...
python quick_check.py

REM 设置 API Key 提示
echo.
echo [3/4] 配置环境...
if "%DASHSCOPE_API_KEY%"=="" (
    echo 提示：未设置 DASHSCOPE_API_KEY
    echo 请设置环境变量：set DASHSCOPE_API_KEY=你的API密钥
) else (
    echo DASHSCOPE_API_KEY 已设置
)

REM 启动应用
echo.
echo [4/4] 启动 Web 应用...
echo.
echo ========================================
echo 应用启动中...
echo 浏览器将自动打开 http://localhost:8501
echo ========================================
echo.
streamlit run app.py

pause
