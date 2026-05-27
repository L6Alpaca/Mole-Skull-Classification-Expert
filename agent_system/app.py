"""
鼹鼠分类智能体 - 主界面
"""
import streamlit as st
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from agent import get_agent


def init_session_state():
    """初始化会话状态"""
    if "agent" not in st.session_state:
        st.session_state["agent"] = get_agent()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "你好！我是鼹鼠分类专家助手。请上传四视角图像，我会帮你识别并解答相关问题。"}
        ]

    if "classification_result" not in st.session_state:
        st.session_state["classification_result"] = None

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = None


def main():
    st.set_page_config(
        page_title="鼹鼠分类智能体",
        page_icon="🐹",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 初始化
    init_session_state()
    agent = st.session_state["agent"]

    # 侧边栏
    with st.sidebar:
        st.title("🐹 鼹鼠分类智能体")
        st.divider()

        st.subheader("📤 图像上传")
        uploaded_files = st.file_uploader(
            "请上传四视角图像（支持 jpg, jpeg, png）",
            type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        if uploaded_files:
            st.session_state["uploaded_files"] = uploaded_files
            st.success(f"已上传 {len(uploaded_files)} 张图像")

            # 显示上传的图像预览
            st.subheader("🖼️ 图像预览")
            cols = st.columns(2)
            for i, img_file in enumerate(uploaded_files):
                with cols[i % 2]:
                    st.image(img_file, caption=f"图像 {i+1}", use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 开始识别", type="primary", use_container_width=True):
                if st.session_state["uploaded_files"]:
                    with st.spinner("正在分析图像..."):
                        # 上传图像到临时目录
                        upload_dir = agent.upload_images(st.session_state["uploaded_files"])
                        # 执行分类
                        result = agent.classify(upload_dir)
                        st.session_state["classification_result"] = result

                        # 更新对话
                        if result["status"] == "success":
                            # 根据是否识别到种级别来显示不同的消息
                            if result["species"]:
                                msg = f"识别完成！该个体属于 **{result['genus']}** 属，**{result['species']}** 种。你可以向我提问关于这个物种的问题。"
                            else:
                                msg = f"识别完成！该个体属于 **{result['genus']}** 属。你可以向我提问关于这个属的问题。"
                        else:
                            msg = f"识别失败：{result['message']}"

                        st.session_state["messages"].append(
                            {"role": "assistant", "content": msg}
                        )
                else:
                    st.warning("请先上传图像")

        with col2:
            if st.button("🗑️ 清空会话", use_container_width=True):
                agent.clear_session()
                st.session_state["uploaded_files"] = None
                st.session_state["classification_result"] = None
                st.session_state["messages"] = [
                    {"role": "assistant", "content": "你好！我是鼹鼠分类专家助手。请上传四视角图像，我会帮你识别并解答相关问题。"}
                ]
                st.rerun()

        st.divider()

        # 显示当前状态
        if st.session_state["classification_result"]:
            result = st.session_state["classification_result"]
            st.subheader("📊 当前识别结果")
            if result["status"] == "success":
                st.success(f"属名：{result['genus']}")
                if result["species"]:
                    st.success(f"种名：{result['species']}")
            else:
                st.error(result["message"])

        # 显示所有支持的类别
        with st.expander("📋 支持的分类类别"):
            class_names = agent.get_class_names()
            for name in class_names:
                st.write(f"- {name}")

    # 主界面 - 对话区域
    st.title("鼹鼠分类专家对话")
    st.divider()

    # 显示对话历史
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 输入区域
    if prompt := st.chat_input("请输入你的问题..."):
        # 添加用户消息
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response_container = st.empty()
                full_response = ""
                try:
                    # 流式输出
                    for chunk in agent.ask_stream(prompt):
                        full_response += chunk
                        response_container.markdown(full_response + "▌")
                    response_container.markdown(full_response)
                except Exception as e:
                    full_response = f"抱歉，发生了错误：{str(e)}"
                    response_container.markdown(full_response)

        st.session_state["messages"].append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    main()
