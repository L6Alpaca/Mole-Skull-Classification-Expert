"""
向量存储服务模块
"""
from langchain_chroma import Chroma
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config


class VectorStoreService:
    """向量存储服务"""

    def __init__(self, embedding):
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embedding,
            persist_directory=config.VECTOR_DB_DIR,
        )

    def get_retriever(self):
        """返回向量检索器"""
        return self.vector_store.as_retriever(
            search_kwargs={"k": config.SIMILARITY_THRESHOLD}
        )


if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings

    print("测试向量检索服务...")
    vector_service = VectorStoreService(
        DashScopeEmbeddings(model=config.EMBEDDING_MODEL_NAME)
    )
    retriever = vector_service.get_retriever()

    print("测试检索...")
    test_query = "长吻鼹的特征"
    results = retriever.invoke(test_query)
    print(f"检索到 {len(results)} 个相关文档")
    for i, doc in enumerate(results):
        print(f"\n文档 {i+1}:")
        print(f"内容: {doc.page_content[:100]}...")
        print(f"来源: {doc.metadata.get('source', '未知')}")
