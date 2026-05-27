@echo off
chcp 65001 >nul
echo ========================================
echo 鼹鼠分类智能体 Web 界面
echo 使用 ai conda 环境启动
echo ========================================
echo.

echo [1/3] 激活 ai 环境...
call conda activate ai
if errorlevel 1 (
    echo 错误：无法激活 ai 环境
    echo 请确保已安装 conda 并创建了 ai 环境
    pause
    exit /b 1
)

echo [2/3] 检查依赖...
python -c "import torch; print('  PyTorch:', torch.__version__)"
python -c "import streamlit; print('  Streamlit:', streamlit.__version__)"
python -c "import numpy; print('  NumPy:', numpy.__version__)"

echo [3/3] 启动 Web 应用...
echo.
echo 请在浏览器中打开: http://localhost:8501
echo 按 Ctrl+C 停止服务
echo.
echo ========================================
streamlit run app.py

pause
