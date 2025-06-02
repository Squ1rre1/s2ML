"""
Wikification 및 Topic Modeling 모듈
텍스트에서 Wikipedia 개념을 추출하고 토픽을 모델링
"""
import re
import requests
import wikipedia
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class Wikifier:
    """텍스트에서 Wikipedia 개념을 추출하고 토픽을 모델링하는 클래스"""
    
    def __init__(self, language='ko'):
        """
        Args:
            language: 처리할 언어 ('ko' 또는 'en')
        """
        self.language = language
        wikipedia.set_lang(language)
        
        # 불용어 설정
        if language == 'ko':
            # 한국어 불용어 (기본적인 것들)
            self.stop_words = set(['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', 
                                 '도', '를', '으로', '자', '에', '와', '한', '하다', '그', '저', 
                                 '것', '수', '등', '년', '월', '일', '때', '및', '또는', '더'])
        else:
            self.stop_words = set(stopwords.words('english'))
            
        self.tfidf_vectorizer = None
        self.lda_model = None
        
    def extract_entities(self, text: str) -> List[str]:
        """
        텍스트에서 주요 엔티티(명사구) 추출
        
        Args:
            text: 입력 텍스트
            
        Returns:
            entities: 추출된 엔티티 리스트
        """
        # 기본적인 전처리
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 토큰화
        tokens = word_tokenize(text.lower())
        
        # 불용어 제거 및 길이 필터링
        entities = [token for token in tokens 
                   if token not in self.stop_words and len(token) > 2]
        
        # n-gram 생성 (bigram, trigram)
        bigrams = [f"{entities[i]} {entities[i+1]}" 
                  for i in range(len(entities)-1)]
        trigrams = [f"{entities[i]} {entities[i+1]} {entities[i+2]}" 
                   for i in range(len(entities)-2)]
        
        all_entities = entities + bigrams + trigrams
        
        return all_entities
    
    def search_wikipedia_concepts(self, entities: List[str], max_concepts: int = 10) -> Dict[str, float]:
        """
        엔티티를 Wikipedia 개념으로 매핑
        
        Args:
            entities: 엔티티 리스트
            max_concepts: 최대 개념 수
            
        Returns:
            concepts: Wikipedia 개념과 관련도 점수
        """
        concept_scores = {}
        entity_counts = Counter(entities)
        
        for entity, count in entity_counts.most_common(max_concepts * 2):
            try:
                # Wikipedia 검색
                search_results = wikipedia.search(entity, results=3)
                
                for result in search_results:
                    try:
                        # 페이지 정보 가져오기
                        page = wikipedia.page(result, auto_suggest=False)
                        
                        # 개념 점수 계산 (빈도 * 검색 순위 가중치)
                        rank_weight = 1.0 / (search_results.index(result) + 1)
                        score = count * rank_weight
                        
                        if page.title in concept_scores:
                            concept_scores[page.title] += score
                        else:
                            concept_scores[page.title] = score
                            
                    except wikipedia.exceptions.DisambiguationError:
                        continue
                    except wikipedia.exceptions.PageError:
                        continue
                        
            except Exception as e:
                logger.debug(f"Error searching for entity '{entity}': {str(e)}")
                continue
        
        # 상위 개념 선택
        sorted_concepts = sorted(concept_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return dict(sorted_concepts[:max_concepts])
    
    def extract_topics_lda(self, documents: List[str], n_topics: int = 5, 
                          n_words: int = 10) -> Tuple[List[List[Tuple[str, float]]], np.ndarray]:
        """
        LDA를 사용한 토픽 모델링
        
        Args:
            documents: 문서 리스트
            n_topics: 토픽 수
            n_words: 각 토픽당 단어 수
            
        Returns:
            topics: 각 토픽의 (단어, 가중치) 리스트
            doc_topic_dist: 문서-토픽 분포
        """
        # TF-IDF 벡터화
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                tokenizer=word_tokenize,
                stop_words=list(self.stop_words)
            )
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        except ValueError:
            # 문서가 너무 적거나 어휘가 부족한 경우
            logger.warning("Not enough documents or vocabulary for LDA")
            return [], np.array([])
        
        # LDA 모델 학습
        if self.lda_model is None:
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
        
        doc_topic_dist = self.lda_model.fit_transform(tfidf_matrix)
        
        # 각 토픽의 상위 단어 추출
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_word_indices = topic.argsort()[-n_words:][::-1]
            top_words = [(feature_names[i], topic[i]) for i in top_word_indices]
            topics.append(top_words)
        
        return topics, doc_topic_dist
    
    def process_segments(self, segments: List[Dict], n_topics: int = 5) -> List[Dict]:
        """
        세그먼트별로 Wikification과 토픽 모델링 수행
        
        Args:
            segments: 텍스트 세그먼트 리스트
            n_topics: 토픽 수
            
        Returns:
            processed_segments: 처리된 세그먼트 (Wikipedia 개념과 토픽 포함)
        """
        processed = []
        documents = [seg['text'] for seg in segments]
        
        # 전체 문서에 대한 토픽 모델링
        topics, doc_topic_dist = self.extract_topics_lda(documents, n_topics)
        
        for idx, segment in enumerate(segments):
            # 엔티티 추출
            entities = self.extract_entities(segment['text'])
            
            # Wikipedia 개념 매핑
            wiki_concepts = self.search_wikipedia_concepts(entities)
            
            # 세그먼트의 주요 토픽
            segment_topics = []
            if len(doc_topic_dist) > idx:
                topic_probs = doc_topic_dist[idx]
                for topic_idx, prob in enumerate(topic_probs):
                    if prob > 0.1:  # 임계값 이상의 토픽만
                        segment_topics.append({
                            'topic_id': topic_idx,
                            'probability': float(prob),
                            'keywords': [(word, float(score)) for word, score in topics[topic_idx][:5]]
                        })
            
            processed_segment = {
                **segment,
                'entities': entities[:20],  # 상위 20개 엔티티
                'wiki_concepts': wiki_concepts,
                'topics': segment_topics
            }
            
            processed.append(processed_segment)
        
        return processed
    
    def calculate_topic_distribution(self, segments: List[Dict]) -> np.ndarray:
        """
        세그먼트별 토픽 분포 계산
        
        Args:
            segments: 처리된 세그먼트 리스트
            
        Returns:
            topic_dist: 세그먼트 x 토픽 분포 행렬
        """
        if not segments or 'topics' not in segments[0]:
            return np.array([])
        
        # 전체 토픽 수 확인
        n_topics = max([topic['topic_id'] 
                       for seg in segments 
                       for topic in seg.get('topics', [])] + [0]) + 1
        
        n_segments = len(segments)
        topic_dist = np.zeros((n_segments, n_topics))
        
        for i, segment in enumerate(segments):
            for topic in segment.get('topics', []):
                topic_dist[i, topic['topic_id']] = topic['probability']
        
        # 정규화
        row_sums = topic_dist.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
        topic_dist = topic_dist / row_sums
        
        return topic_dist 