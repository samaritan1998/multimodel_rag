from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from pdf2image import convert_from_path

class DocumentQA:
    """
    一个用于基于检索增强生成模型和视觉语言模型进行文档问答的类。
    """

    def __init__(self, rag_model_name: str, vlm_model_name: str, device: str = 'cuda', system_prompt: str = None):
        """
        初始化 DocumentQA 实例。

        参数:
        - rag_model_name (str): 使用的RAG模型名称。
        - vlm_model_name (str): 使用的视觉语言模型名称。
        - device (str): 模型运行的设备（'cuda' 或 'cpu'）。
        - system_prompt (str, optional): 对话中使用的系统提示词。
        """
        # 初始化 RAG 引擎
        self.rag_engine = RAGMultiModalModel.from_pretrained(rag_model_name)

        # 初始化视觉语言模型（VLM）
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device
        )

        # 初始化处理器
        self.processor = AutoProcessor.from_pretrained(vlm_model_name, trust_remote_code=True)
        self.device = device

        # 设置系统提示词
        if system_prompt is None:
            self.system_prompt = (
                "你是一位专精于计算机科学和机器学习的AI研究助理。"
                "你的任务是分析学术论文，尤其是关于文档检索和多模态模型的研究。"
                "请仔细分析提供的图像和文本，提供深入的见解和解释。"
            )
        else:
            self.system_prompt = system_prompt

    def index_document(self, pdf_path: str, index_name: str = 'index', overwrite: bool = True):
        """
        索引指定的 PDF 文档。

        参数:
        - pdf_path (str): PDF 文件的路径。
        - index_name (str): 创建的索引名称。
        - overwrite (bool): 是否覆盖现有索引。
        """
        self.pdf_path = pdf_path
        self.rag_engine.index(
            input_path=pdf_path,
            index_name=index_name,
            store_collection_with_index=False,
            overwrite=overwrite
        )

        # 将 PDF 页转换为图像
        self.images = convert_from_path(pdf_path)

    def query(self, text_query: str, k: int = 3) -> str:
        """
        使用文本问题查询文档。

        参数:
        - text_query (str): 要询问的问题。
        - k (int): 要检索的搜索结果数。

        返回:
        - str: 生成的答案。
        """
        # 使用 RAG 引擎进行搜索
        results = self.rag_engine.search(text_query, k=k)
        print("搜索结果:", results)

        # 处理当没有搜索结果时的情况
        if not results:
            print("未找到相关查询结果。")
            return None

        # 从结果中获取页面编号
        try:
            page_num = results[0]["page_num"]
            image_index = page_num - 1  # 调整为从 0 开始的索引
            image = self.images[image_index]
        except (KeyError, IndexError) as e:
            print("获取页面图像时出错:", e)
            return None

        # 准备对话消息
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_query},
                ],
            }
        ]

        # 应用聊天模板
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)

        # 准备模型输入
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 生成输出
        generated_ids = self.vlm.generate(**inputs, max_new_tokens=1024)

        # 对输出进行后处理
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 返回生成的文本
        return output_text[0]  # 假设批量大小为 1

if __name__ == "__main__":
    # 初始化 DocumentQA 实例
    document_qa = DocumentQA(
        rag_model_name="vidore/colpali",
        vlm_model_name="Qwen/Qwen2-VL-7B-Instruct",
        device='cuda'
    )

    # 索引 PDF 文档
    document_qa.index_document("colpali.pdf")

    # 定义查询
    text_query = (
        "ColPali 模型在哪个数据集上相比其他模型有最大的优势？"
        "该优势的改进幅度是多少？"
    )

    # 执行查询并打印答案
    answer = document_qa.query(text_query)
    print("答案:", answer)
