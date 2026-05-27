"""
对话历史存储模块
"""
import os
import json
from pathlib import Path
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

import config


class FileChatMessageHistory(BaseChatMessageHistory):
    """
    基于文件存储的对话历史
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._messages: List[BaseMessage] = []
        self._load_from_file()

    def _load_from_file(self):
        """从文件加载对话历史"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._messages = messages_from_dict(data)
            except Exception:
                self._messages = []

    def _save_to_file(self):
        """保存对话历史到文件"""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(messages_to_dict(self._messages), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存对话历史失败: {e}")

    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._save_to_file()

    def clear(self) -> None:
        self._messages = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


def get_history(session_id: str):
    """
    获取指定session的对话历史
    """
    history_dir = Path(__file__).parent / "chat_history"
    history_dir.mkdir(exist_ok=True)
    file_path = str(history_dir / f"{session_id}.json")
    return FileChatMessageHistory(file_path)


if __name__ == "__main__":
    # 测试
    history = get_history("test_session")
    from langchain_core.messages import HumanMessage, AIMessage

    history.add_message(HumanMessage(content="你好"))
    history.add_message(AIMessage(content="你好，有什么可以帮助你？"))

    print("对话历史:")
    for msg in history.messages:
        print(f"{msg.type}: {msg.content}")
