"""
RAG服务模块 - 结合图像分类结果和知识库进行问答
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableWithMessageHistory,
    RunnableLambda
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import config
from vector_stores import VectorStoreService
from file_history_store import get_history


def print_prompt(prompt):
    """打印prompt用于调试"""
    print("="*30)
    print(prompt.to_string())
    print("="*30)
    return prompt


class RagService:
    """RAG服务 - 支持结合图像分类结果的问答"""

    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.EMBEDDING_MODEL_NAME)
        )

        # 系统提示词 - 专门针对鼹鼠分类和知识库问答
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个专业的鼹鼠分类专家助手。"),
                ("system", "以下是系统识别出的图像分类结果：{classification_result}"),
                ("system", "请结合以下参考资料回答用户问题。参考资料:{context}。"),
                ("system", "用户的对话历史记录如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(
            model=config.CHAT_MODEL_NAME,
            streaming=True
        )

        self.chain = self.__get_chain()
        self.current_classification = None

    def set_classification(self, classification_result):
        """
        设置当前的分类结果
        :param classification_result: 图像分类结果字典
        """
        self.current_classification = classification_result

    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            """格式化检索到的文档"""
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str

        def format_classification(classification_result):
            """格式化分类结果"""
            if classification_result is None:
                return "暂无图像分类结果"

            if classification_result.get("status") == "success":
                genus = classification_result.get("genus", "未知")
                species = classification_result.get("species")
                if species:
                    return f"识别结果：该个体属于 {genus} 属，{species} 种"
                else:
                    return f"识别结果：该个体属于 {genus} 属"
            else:
                return f"识别失败：{classification_result.get('message', '未知错误')}"

        def build_chain_input(value: dict) -> dict:
            """构建最终的输入字典"""
            return {
                "input": value["input"]["input"],
                "context": value["context"],
                "history": value["input"]["history"],
                "classification_result": format_classification(self.current_classification)
            }

        def get_input_for_retriever(value: dict) -> str:
            """获取用于检索的输入"""
            base_input = value["input"]
            if self.current_classification and self.current_classification.get("status") == "success":
                # 优先使用species进行检索，如果没有species则使用genus
                species = self.current_classification.get("species")
                if species:
                    return f"{species} {base_input}"
                genus = self.current_classification.get("genus", "")
                return f"{genus} {base_input}"
            return base_input

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": (
                    RunnableLambda(get_input_for_retriever)
                    | retriever
                    | format_document
                )
            }
            | RunnableLambda(build_chain_input)
            | self.prompt_template
            # | print_prompt  # 调试时开启
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

    def ask(self, question, session_id=None):
        """
        提问
        :param question: 用户问题
        :param session_id: 会话ID，用于保存对话历史
        :return: AI回答
        """
        if session_id is None:
            session_id = config.SESSION_CONFIG["configurable"]["session_id"]

        session_config = {
            "configurable": {
                "session_id": session_id
            }
        }

        return self.chain.invoke({"input": question}, session_config)

    def ask_stream(self, question, session_id=None):
        """
        流式提问
        :param question: 用户问题
        :param session_id: 会话ID
        :yield: AI回答的片段
        """
        if session_id is None:
            session_id = config.SESSION_CONFIG["configurable"]["session_id"]

        session_config = {
            "configurable": {
                "session_id": session_id
            }
        }

        return self.chain.stream({"input": question}, session_config)

    def clear_history(self, session_id=None):
        """清空对话历史"""
        if session_id is None:
            session_id = config.SESSION_CONFIG["configurable"]["session_id"]
        history = get_history(session_id)
        history.clear()


if __name__ == '__main__':
    # 简单测试
    print("测试RAG服务...")
    rag = RagService()

    # 设置分类结果
    test_classification = {
        "genus": "Euroscaptor",
        "species": "Euroscaptor longirostris",
        "class_idx": 0,
        "status": "success",
        "message": "分类成功"
    }
    rag.set_classification(test_classification)

    print(f"设置分类结果: {test_classification}")
    print("提问: 这个种的鼹鼠有什么特点？")

    response = rag.ask("这个种的鼹鼠有什么特点？")
    print(f"回答: {response}")