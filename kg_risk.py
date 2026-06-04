# kg_risk.py
from arango import ArangoClient
import re
import torch
import numpy as np
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import json


def extract_text_content(text):
    result = {
        'mlc_target': [],
        'ner_target': {"电压等级":[], "设备名称":[], "工作内容":[]},
        're_target': []
    }
    
    # 提取多标签分类内容
    mlc_pattern = r'(.*?)(?=实体识别:|关系抽取:|$)'
    mlc_match = re.search(mlc_pattern, text)
    if mlc_match:
        risk_text = mlc_match.group(1).strip()
        risks = risk_text.strip("<>").split("><")
        result['mlc_target'] = risks
    
    # 提取实体识别内容
    ner_pattern = r'实体识别:(.*?)(?=关系抽取:|$)'
    ner_match = re.search(ner_pattern, text)
    if ner_match:
        ner_content = ner_match.group(1)
        
        # 使用正则表达式一次性匹配： (标签)值1;值2;值3;
        # 匹配模式：括号内的标签，后面跟着多个用分号分隔的值，直到遇到下一个(或结束
        pattern = r'\(([^)]+)\)([^()]+?)(?=\(|$)'
        
        for match in re.finditer(pattern, ner_content):
            label = match.group(1)
            values_text = match.group(2).strip()
            
            # 按分号分割值，并过滤空值
            if values_text:
                values = [v.strip() for v in values_text.split(';') if v.strip()]
                for value in values:
                    result['ner_target'][label].append(value)
                    # result['ner_target'].append(f"{value}@{label}")
    
    # 提取关系抽取内容
    re_pattern = r'关系抽取:(.*?)$'
    re_match = re.search(re_pattern, text)
    if re_match:
        re_content = re_match.group(1)
        re_text = re_content[4:-1]
        result['re_target'] = re_text.split(";")
    
    return result['mlc_target'], result['ner_target'], result['re_target']

class KGRiskPredictor:
    """基于知识图谱的风险预测器"""
    
    def __init__(self, host: str, db_name: str, username: str, password: str):
        self.client = ArangoClient(hosts=host)
        self.db = self.client.db(db_name, username=username, password=password)
        
        knowledge_path = "./knowledge.json"
        extract_model_path = "/home/hh/seq2seq/mengzi-t5/mix_10_multi_10/best_model_score_0.9138"
        llm_model_path = "/home/hh/MODELS/Qwen3-8B"
        embedding_model_path = "/disk3/lsp/Power-Safety-Knowledge-Graph-Application-System/sentence-transformers/all-mpnet-base-v2"
        
        # 读取知识
        self.knowledge = self.load_knowledge_from_graph(knowledge_path)
        print("成功读取知识图谱!")

        # 加载知识抽取模型
        self.ex_tokenizer = AutoTokenizer.from_pretrained(extract_model_path,local_files_only=True)
        self.extractor = AutoModelForSeq2SeqLM.from_pretrained(extract_model_path,local_files_only=True)
        print("成功加载知识抽取模型mengzi-t5!")

        # 加载LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        # Qwen 必须设置 pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        print("成功加载大模型!")

        # 加载 SentenceTransformer 并预计算知识库向量
        self.embedder = None
        self.knowledge_embeddings = None
        try:
            self.embedder = SentenceTransformer(embedding_model_path)
            self._build_knowledge_embeddings()
            print("成功加载向量编码器并预计算知识库向量!")
        except Exception as e:
            print(f"[WARNING] SentenceTransformer 加载失败，将使用 Jaccard fallback: {e}")

    def load_knowledge_from_graph(self, file_path):
        "加载知识图谱中的知识"
        with open(file_path, "r") as f:
            datas = json.load(f)
        return datas

    def _encode_knowledge_text(self, know_item):
        """将一条知识的实体组和风险组拼接为可编码的中文文本"""
        text = know_item.get("文本", "")
        entities = know_item.get("实体组", {})
        risks = know_item.get("风险组", [])

        parts = [f"文本：{text}"]
        for key in ["电压等级", "设备名称", "工作内容"]:
            values = entities.get(key, [])
            if values:
                parts.append(f"{key}：{'、'.join(values)}")
        if risks:
            parts.append(f"风险：{'、'.join(risks)}")
        return "；".join(parts)

    def _build_knowledge_embeddings(self):
        """启动时批量编码整个 knowledge 列表，缓存为 numpy 数组"""
        texts = [self._encode_knowledge_text(know) for know in self.knowledge]
        self.knowledge_embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"预计算完成，知识库向量维度: {self.knowledge_embeddings.shape}")

    def _get_candidates_by_vector(self, query_text, topk):
        """向量余弦相似度检索，返回与 Jaccard 相同结构的候选列表"""
        query_embedding = self.embedder.encode([query_text], convert_to_numpy=True)[0]

        # cosine similarity（mpnet 输出已近似归一化）
        similarities = np.dot(self.knowledge_embeddings, query_embedding)
        # 归一化保证 cosine 范围在 [-1, 1]
        norms = np.linalg.norm(self.knowledge_embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = similarities / np.clip(norms, a_min=1e-8, a_max=None)

        top_indices = np.argsort(similarities)[::-1][:topk]
        return [
            {
                "knowledge": self.knowledge[int(idx)],
                "score": round(float(similarities[idx]), 4)
            }
            for idx in top_indices
        ]

    def extract_entities(self, text: str, max_length: int = 128):
        """从文本中抽取实体"""
        instruction = "<坠落><外力外物致伤><触电><烧、烫伤><中毒><窒息><减供负荷><电能质量不稳定><系统失稳><非正常解列><电力监控系统网络破坏><设备损坏><设备性能下降><被迫停运>实体识别:(电压等级)(设备名称)(工作内容)关系抽取:[修饰]"
        prompt = text + instruction
        # 输入编码
        inputs = self.ex_tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # 生成预测
        outputs = self.extractor.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=0,
            length_penalty=1.0,
            do_sample=False,
            pad_token_id=self.ex_tokenizer.pad_token_id,
            eos_token_id=self.ex_tokenizer.eos_token_id,
            bad_words_ids=[[32127], [0]] if self.ex_tokenizer.pad_token_id == 0 else [[32127]]
        )

        # 解码输出
        result = self.ex_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(result)
        pred_mlc, pred_ner, pred_re = extract_text_content(result)
        output = {
            "文本": text,
            "实体组": pred_ner,
            "风险组": pred_mlc
        }
        # print(json.dumps(output, ensure_ascii=False, indent=2))
        return output

    def _convert_dict_to_list(self, key_dict):
        key_list = []
        for k, v in key_dict.items():
            for e in v:
                key_list.append(e + "@" + k)
        return key_list

    def calculate_list_similarity(self, list_a, list_b):
        """
        计算两个列表的非空元素重叠度（Jaccard 相似度）
        交集大小 / 并集大小
        """
        # 过滤空字符串，转成集合
        set_a = set(item.strip() for item in list_a if item.strip())
        set_b = set(item.strip() for item in list_b if item.strip())

        # 空集合直接返回0
        if not set_a or not set_b:
            return 0.0

        # 计算 Jaccard
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    def _get_candidates_by_jaccard(self, input_dict, topk):
        """原有 Jaccard 检索，保留作为 fallback"""
        query = self._convert_dict_to_list(input_dict["实体组"])

        scored = []
        for know in self.knowledge:
            cand = self._convert_dict_to_list(know["实体组"])
            score = self.calculate_list_similarity(query, cand)
            scored.append({
                "knowledge": know,
                "score": round(score, 4)
            })

        scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
        return scored_sorted[:topk]

    def get_candidates(self, input_dict, topk, retrieval_method="vector"):
        """检索入口：根据 retrieval_method 选择向量或 Jaccard 检索"""
        if retrieval_method == "vector" and self.embedder is not None and self.knowledge_embeddings is not None:
            try:
                query_text = self._encode_knowledge_text(input_dict)
                results = self._get_candidates_by_vector(query_text, topk)
                if results and results[0]["score"] > 0:
                    print(f"[检索] 向量检索成功，top-1 score: {results[0]['score']}")
                    return results
                else:
                    print("[检索] 向量检索返回空结果，回退 Jaccard")
            except Exception as e:
                print(f"[检索] 向量检索失败，回退 Jaccard: {e}")

        print("[检索] 使用 Jaccard 检索")
        return self._get_candidates_by_jaccard(input_dict, topk)

    def llm_predict(self, messages):
        # 1. 模型推理
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1500
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,  # 固定输出，更稳定
            )

        # 2. 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        # 3. 解析列表（确保输出是列表）
        try:
            start = response.find("[")
            text = response[start:]
            risk_list = json.loads(text)
            if not isinstance(risk_list, list):
                return []
            return risk_list
        except:
            return []

    def predict_risks(self, input_text: str, top_k: int = 5, retrieval_method: str = "vector", max_matched_texts: int = 10) -> Dict:
        """基于知识图谱预测风险"""

        # Step 1: 实体抽取
        result = self.extract_entities(input_text)
        print("\n实体抽取结果")
        print(result)

        # Step 2: 获取知识候选集
        knowledge_candidates = self.get_candidates(result, top_k, retrieval_method=retrieval_method)
        print("\n候选知识示例")
        print(knowledge_candidates[:3])

        # Step 3: LLM 对实体组打分


        # Step 4: LLM 根据知识图谱预测风险
        query_text = json.dumps(result, ensure_ascii=False, indent=2)
        knowledge_text = ""
        for i, item in enumerate(knowledge_candidates, 1):
            know_dict = {"文本":item["knowledge"]["文本"], "实体组":item["knowledge"]["实体组"]}
            know_text = json.dumps(know_dict, ensure_ascii=False, indent=2)
            risk = item["knowledge"]["风险组"]
            knowledge_text += f"参考{i}：{know_text} → 风险：{risk}\n"
        
        prompt = f"""
        你是电力安全风险识别专家。根据用户输入的文本和预测关键字，结合参考案例，预测可能发生的风险。

        输入文本和关键字：{query_text}

        参考相似案例：
        {knowledge_text}

        请严格只输出风险列表，格式如下：
        ["风险1", "风险2"]

        不要解释、不要多余文字、不要JSON，只输出列表。/no_think
        """
        print(f"\nprompt长度:{len(prompt)}")

        messages = [
            {"role": "system", "content": "你是专业的电力安全风险预测专家。"},
            {"role": "user", "content": prompt.strip()}
        ]
        response = self.llm_predict(messages)
        
        final_result = result
        final_result["大模型预测"] = response
        print("\nLLM预测结果")
        print(final_result)
        return knowledge_candidates, final_result


# # ArangoDB连接配置
# ARANGO_HOST = 'http://localhost:8529'
# ARANGO_USERNAME = 'root'
# ARANGO_PASSWORD = 'root'
# ARANGO_DB_NAME = 'power safety'

# kg_predictor = KGRiskPredictor(
#     host=ARANGO_HOST,
#     db_name=ARANGO_DB_NAME,
#     username=ARANGO_USERNAME,
#     password=ARANGO_PASSWORD
# )

# kg_predictor.predict_risks("220kv七星站退出110kv5m、6m母差保护。")