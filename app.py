from flask import Flask, render_template, request, jsonify
from arango import ArangoClient
from kg_risk import KGRiskPredictor
import re
import json
from datetime import datetime

app = Flask(__name__)

# ArangoDB连接配置
ARANGO_HOST = 'http://localhost:8529'
ARANGO_USERNAME = 'root'
ARANGO_PASSWORD = 'root'
ARANGO_DB_NAME = 'power safety'

# 全局知识图谱预测器
kg_predictor = None

def get_db():
    """获取数据库连接"""
    client = ArangoClient(hosts=ARANGO_HOST)
    db = client.db(ARANGO_DB_NAME, username=ARANGO_USERNAME, password=ARANGO_PASSWORD)
    return db

def init_kg_predictor():
    """初始化知识图谱预测器"""
    global kg_predictor
    if kg_predictor is None:
        kg_predictor = KGRiskPredictor(
            host=ARANGO_HOST,
            db_name=ARANGO_DB_NAME,
            username=ARANGO_USERNAME,
            password=ARANGO_PASSWORD
        )
    return kg_predictor

def insert_entity(db, collection_name, entity_content):
    """插入实体到指定集合，如果已存在则返回现有实体"""
    # 检查实体是否已存在
    check_query = f"""
    FOR entity IN `{collection_name}`
        FILTER entity.content == @content
        RETURN entity
    """
    cursor = db.aql.execute(check_query, bind_vars={'content': entity_content})
    existing = [doc for doc in cursor]
    
    if existing:
        return existing[0]
    
    # 插入新实体
    entity = {
        'content': entity_content,
        'type': collection_name,
        'created_at': datetime.now().isoformat()
    }
    result = db.collection(collection_name).insert(entity)
    return {**entity, '_key': result['_key'], '_id': result['_id']}

def insert_text_node(db, text_content, source='预训练模型'):
    """插入文本节点，如果已存在则返回现有文本"""
    check_query = """
    FOR doc IN `文本`
        FILTER doc.content == @content
        RETURN doc
    """
    cursor = db.aql.execute(check_query, bind_vars={'content': text_content})
    existing = [doc for doc in cursor]
    
    if existing:
        return existing[0]
    
    # 插入新文本
    text_node = {
        'content': text_content,
        'source': source,
        'created_at': datetime.now().isoformat()
    }
    result = db.collection('文本').insert(text_node)
    return {**text_node, '_key': result['_key'], '_id': result['_id']}

def insert_edge(db, edge_collection, from_id, to_id):
    """插入边，如果已存在则跳过"""
    check_query = f"""
    FOR edge IN `{edge_collection}`
        FILTER edge._from == @from_id AND edge._to == @to_id
        RETURN edge
    """
    cursor = db.aql.execute(check_query, bind_vars={'from_id': from_id, 'to_id': to_id})
    existing = [doc for doc in cursor]
    
    if existing:
        return existing[0]
    
    # 插入新边
    edge = {
        '_from': from_id,
        '_to': to_id,
        'created_at': datetime.now().isoformat()
    }
    result = db.collection(edge_collection).insert(edge)
    return {**edge, '_key': result['_key'], '_id': result['_id']}

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


# ==================== 模块1：知识图谱检索 - 文本检索 ====================

@app.route('/api/search/text', methods=['POST'])
def search_text():
    """
    文本检索功能
    根据输入文本，检索文本集合中 content 属性匹配的节点
    返回文本节点信息，以及以该文本为头节点的所有边
    """
    data = request.get_json()
    input_content = data.get('content', '').strip()
    
    if not input_content:
        return jsonify({'success': False, 'error': '请输入文本内容'})
    
    db = get_db()
    
    try:
        # 第一步：查找匹配的文本节点
        text_query = """
        FOR doc IN `文本`
            FILTER doc.content == @input_content
            LIMIT 1
            RETURN doc
        """
        
        text_cursor = db.aql.execute(text_query, bind_vars={'input_content': input_content})
        text_nodes = [doc for doc in text_cursor]
        
        if not text_nodes:
            return jsonify({'success': False, 'error': f'未找到匹配的文本: {input_content}'})
        
        text_node = text_nodes[0]
        text_id = text_node['_id']
        
        # 第二步：查询以该文本为源节点的所有边
        edges_query = """
        LET all_edges = (
            FOR edge IN `关键字`
                FILTER edge._from == @text_id
                RETURN {
                    edge_key: edge._key,
                    edge_id: edge._id,
                    edge_type: '关键字',
                    from: edge._from,
                    to: edge._to,
                    to_entity: DOCUMENT(edge._to)
                }
        )
        
        LET cause_edges = (
            FOR edge IN `导致`
                FILTER edge._from == @text_id
                RETURN {
                    edge_key: edge._key,
                    edge_id: edge._id,
                    edge_type: '导致',
                    from: edge._from,
                    to: edge._to,
                    to_entity: DOCUMENT(edge._to)
                }
        )
        
        FOR edge IN UNION(all_edges, cause_edges)
            RETURN edge
        """
        
        edges_cursor = db.aql.execute(edges_query, bind_vars={'text_id': text_id})
        edges = [doc for doc in edges_cursor]
        
        return jsonify({
            'success': True,
            'results': {
                'found': True,
                'text_node': {
                    'key': text_node.get('_key', ''),
                    'id': text_node.get('_id', ''),
                    'content': text_node.get('content', ''),
                    'source': text_node.get('source', '')
                },
                'outgoing_edges': edges
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'查询失败: {str(e)}'})


# ==================== 模块1：知识图谱检索 - 关联查询 ====================

@app.route('/api/search/entity_risks', methods=['POST'])
def search_entity_risks():
    """
    关联查询：查询某个实体（电压等级/设备名称/工作内容）涉及的所有危害因素
    通过实体节点 -> 包含该实体的文本节点 -> 文本关联的风险标签
    """
    data = request.get_json()
    entity_type = data.get('entity_type', '设备名称')
    entity_name = data.get('entity_name', '').strip()
    
    if not entity_name:
        return jsonify({'success': False, 'error': '请输入实体名称'})
    
    valid_types = ['电压等级', '设备名称', '工作内容']
    if entity_type not in valid_types:
        return jsonify({'success': False, 'error': f'实体类型必须是: {", ".join(valid_types)}'})
    
    db = get_db()
    
    try:
        # 第一步：查找实体节点
        entity_query = f"""
        FOR entity IN `{entity_type}`
            FILTER entity.content == @entity_name
            LIMIT 1
            RETURN entity
        """
        entity_cursor = db.aql.execute(entity_query, bind_vars={'entity_name': entity_name})
        entities = [doc for doc in entity_cursor]
        
        if not entities:
            return jsonify({'success': False, 'error': f'未找到{entity_type}: {entity_name}'})
        
        entity = entities[0]
        entity_id = entity['_id']
        
        # 第二步：查找包含该实体的文本节点
        texts_query = """
        FOR text IN INBOUND @entity_id
            `关键字`
            RETURN text
        """
        texts_cursor = db.aql.execute(texts_query, bind_vars={'entity_id': entity_id})
        texts = [doc for doc in texts_cursor]
        
        if not texts:
            return jsonify({
                'success': True,
                'results': {
                    'found': True,
                    'entity': {
                        'key': entity.get('_key', ''),
                        'name': entity.get('content', entity_name),
                        'type': entity_type
                    },
                    'related_texts': [],
                    'related_texts_count': 0,
                    'risks': []
                }
            })
        
        # 第三步：查找这些文本关联的风险
        text_ids = [t['_id'] for t in texts]
        
        risks_query = """
        FOR text_id IN @text_ids
            FOR risk IN OUTBOUND text_id
                `导致`
            RETURN DISTINCT risk
        """
        risks_cursor = db.aql.execute(risks_query, bind_vars={'text_ids': text_ids})
        risks = [doc for doc in risks_cursor]
        
        related_texts = []
        for text in texts:
            related_texts.append({
                'key': text.get('_key', ''),
                'id': text.get('_id', ''),
                'content': text.get('content', ''),
                'source': text.get('source', '未知')
            })
        
        return jsonify({
            'success': True,
            'results': {
                'found': True,
                'entity': {
                    'key': entity.get('_key', ''),
                    'name': entity.get('content', entity_name),
                    'type': entity_type
                },
                'related_texts': related_texts,
                'related_texts_count': len(texts),
                'risks': [{
                    'risk_key': r.get('_key', ''),
                    'risk_name': r.get('content', r.get('name', '未知')),
                    'risk_category': r.get('type', '风险')
                } for r in risks]
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'查询失败: {str(e)}'})


# ==================== 模块2：知识图谱构建（预训练模型预测 + 用户确认 + 图谱插入） ====================

@app.route('/api/pretrain_predict', methods=['POST'])
def pretrain_predict():
    """
    基于预训练模型的风险预测接口（仅预测，不插入）
    返回实体识别、关系抽取、风险预测结果，供用户确认
    """
    data = request.get_json()
    input_text = data.get('text', '').strip()
    
    if not input_text:
        return jsonify({'success': False, 'error': '请输入文本内容'})
    
    try:
        predictor = init_kg_predictor()
        
        # 调用预训练模型进行预测
        result = predictor.extract_entities(input_text)
        
        # 提取结果
        extracted_entities = result.get('实体组', {})  # 包含电压等级、设备名称、工作内容
        extracted_risks = result.get('风险组', [])     # 风险列表
        relations = result.get('关系组', [])           # 关系抽取结果（如果有）
        
        # 生成唯一标识
        import hashlib
        session_id = hashlib.md5(f"{input_text}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        # 构建返回的实体和风险数据，为每个项目添加唯一ID
        entities_with_id = {
            '电压等级': [{'id': f"{session_id}_volt_{i}", 'value': v, 'correct': True} for i, v in enumerate(extracted_entities.get('电压等级', []))],
            '设备名称': [{'id': f"{session_id}_dev_{i}", 'value': v, 'correct': True} for i, v in enumerate(extracted_entities.get('设备名称', []))],
            '工作内容': [{'id': f"{session_id}_work_{i}", 'value': v, 'correct': True} for i, v in enumerate(extracted_entities.get('工作内容', []))]
        }
        
        risks_with_id = [{'id': f"{session_id}_risk_{i}", 'value': r, 'correct': True} for i, r in enumerate(extracted_risks)]
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'input_text': input_text,
            'entities': entities_with_id,
            'risks': risks_with_id,
            'relations': relations,
            'original_text': result.get('文本', input_text)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'预训练模型预测失败: {str(e)}'
        })


@app.route('/api/confirm_and_insert', methods=['POST'])
def confirm_and_insert():
    """
    用户确认后，将正确的内容插入知识图谱
    """
    data = request.get_json()
    input_text = data.get('text', '').strip()
    confirmed_entities = data.get('confirmed_entities', {})  # 格式: {'电压等级': ['value1', ...], ...}
    confirmed_risks = data.get('confirmed_risks', [])        # 格式: ['risk1', 'risk2', ...]
    
    if not input_text:
        return jsonify({'success': False, 'error': '请输入文本内容'})
    
    db = get_db()
    
    try:
        insert_results = {
            'text_node': None,
            'entities': {},
            'edges': {
                'keyword_edges': [],
                'cause_edges': []
            },
            'risks': []
        }
        
        # 1. 插入文本节点
        text_node = insert_text_node(db, input_text, source='预训练模型（用户确认）')
        insert_results['text_node'] = {
            'key': text_node.get('_key'),
            'id': text_node.get('_id'),
            'content': text_node.get('content'),
            'is_new': text_node.get('created_at') is not None
        }
        
        text_id = text_node['_id']
        
        # 2. 插入确认的实体节点并创建关键字边
        entity_collections = {
            '电压等级': confirmed_entities.get('电压等级', []),
            '设备名称': confirmed_entities.get('设备名称', []),
            '工作内容': confirmed_entities.get('工作内容', [])
        }
        
        for collection_name, entity_values in entity_collections.items():
            insert_results['entities'][collection_name] = []
            for entity_value in entity_values:
                if entity_value and entity_value.strip():
                    # 插入实体
                    entity_node = insert_entity(db, collection_name, entity_value)
                    insert_results['entities'][collection_name].append({
                        'content': entity_node.get('content'),
                        'key': entity_node.get('_key'),
                        'id': entity_node.get('_id'),
                        'is_new': entity_node.get('created_at') is not None
                    })
                    
                    # 创建关键字边（文本 -> 实体）
                    edge = insert_edge(db, '关键字', text_id, entity_node['_id'])
                    insert_results['edges']['keyword_edges'].append({
                        'from': text_id,
                        'to': entity_node['_id'],
                        'edge_key': edge.get('_key'),
                        'is_new': edge.get('created_at') is not None
                    })
        
        # 3. 插入确认的风险节点并创建导致边
        for risk_value in confirmed_risks:
            if risk_value and risk_value.strip():
                # 插入风险节点
                risk_node = insert_entity(db, '风险', risk_value)
                insert_results['risks'].append({
                    'name': risk_node.get('content'),
                    'key': risk_node.get('_key'),
                    'id': risk_node.get('_id'),
                    'is_new': risk_node.get('created_at') is not None
                })
                
                # 创建导致边（文本 -> 风险）
                edge = insert_edge(db, '导致', text_id, risk_node['_id'])
                insert_results['edges']['cause_edges'].append({
                    'from': text_id,
                    'to': risk_node['_id'],
                    'edge_key': edge.get('_key'),
                    'is_new': edge.get('created_at') is not None
                })
        
        return jsonify({
            'success': True,
            'message': '知识图谱更新成功',
            'insert_results': insert_results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'插入知识图谱失败: {str(e)}'
        })


# ==================== 模块3：风险预警（知识图谱预测） ====================

@app.route('/api/kg_predict', methods=['POST'])
def kg_predict():
    """
    基于知识图谱的风险预测接口
    """
    data = request.get_json()
    input_text = data.get('text', '').strip()
    top_k = data.get('top_k', 5)
    retrieval_method = data.get('retrieval_method', 'vector')

    if not input_text:
        return jsonify({'success': False, 'error': '请输入文本内容'})

    valid_methods = ['vector', 'jaccard']
    if retrieval_method not in valid_methods:
        return jsonify({
            'success': False,
            'error': f'检索方法参数无效，仅支持: {", ".join(valid_methods)}'
        }), 400

    try:
        predictor = init_kg_predictor()
        knowledge_candidates, final_result = predictor.predict_risks(input_text, top_k=top_k, retrieval_method=retrieval_method)
        
        return jsonify({
            'success': True,
            'input_text': input_text,
            'extracted_entities': final_result.get('实体组', {}),
            'extracted_risks': final_result.get('风险组', []),
            'llm_prediction': final_result.get('大模型预测', []),
            'knowledge_candidates': knowledge_candidates,
            'original_text': final_result.get('文本', input_text)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'知识图谱预测失败: {str(e)}'
        })


if __name__ == '__main__':
    init_kg_predictor()
    app.run(debug=True, host='0.0.0.0', port=5000)