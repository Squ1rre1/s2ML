"""
동적 Segmentation 모듈
엔트로피와 JSD를 사용하여 토픽 기반 동적 세그먼테이션 수행
"""
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DynamicSegmenter:
    """토픽 분포 기반 동적 세그먼테이션 클래스"""
    
    def __init__(self, entropy_threshold: float = 0.5, 
                 jsd_threshold: float = 0.3,
                 min_segment_length: int = 3):
        """
        Args:
            entropy_threshold: 엔트로피 변화 임계값
            jsd_threshold: JSD 임계값
            min_segment_length: 최소 세그먼트 길이
        """
        self.entropy_threshold = entropy_threshold
        self.jsd_threshold = jsd_threshold
        self.min_segment_length = min_segment_length
        
    def normalize_distribution(self, distribution: np.ndarray) -> np.ndarray:
        """
        Step 1: 토픽 분포 정규화
        
        Args:
            distribution: 토픽 분포 벡터
            
        Returns:
            normalized: 정규화된 분포
        """
        # 0 방지를 위한 스무딩
        distribution = distribution + 1e-10
        
        # 정규화
        normalized = distribution / np.sum(distribution)
        
        return normalized
    
    def calculate_entropy(self, distribution: np.ndarray) -> float:
        """
        Step 2: 엔트로피 계산
        H_i = -∑(p_ij * log(p_ij))
        
        Args:
            distribution: 정규화된 토픽 분포
            
        Returns:
            entropy_value: 엔트로피 값
        """
        # 0 log 0 = 0 처리
        distribution = np.where(distribution > 0, distribution, 1e-10)
        
        entropy_value = -np.sum(distribution * np.log(distribution))
        
        return entropy_value
    
    def calculate_variation_scores(self, distributions: np.ndarray) -> List[Dict[str, float]]:
        """
        Step 3: 변화 점수 계산
        V_i = {|H_i - H_{i-1}|, JSD(P_{i-1}||P_i)}
        
        Args:
            distributions: 세그먼트별 토픽 분포 (n_segments x n_topics)
            
        Returns:
            variation_scores: 각 위치의 변화 점수
        """
        n_segments = distributions.shape[0]
        variation_scores = []
        
        # 첫 번째 세그먼트는 변화 점수 없음
        variation_scores.append({
            'entropy_diff': 0.0,
            'jsd': 0.0,
            'combined': 0.0
        })
        
        # 각 세그먼트 쌍에 대해 계산
        for i in range(1, n_segments):
            # 분포 정규화
            p_prev = self.normalize_distribution(distributions[i-1])
            p_curr = self.normalize_distribution(distributions[i])
            
            # 엔트로피 계산
            h_prev = self.calculate_entropy(p_prev)
            h_curr = self.calculate_entropy(p_curr)
            
            # 엔트로피 차이
            entropy_diff = abs(h_curr - h_prev)
            
            # Jensen-Shannon Divergence
            jsd_value = jensenshannon(p_prev, p_curr) ** 2  # squared for JSD
            
            # 결합 점수 (두 메트릭의 가중 평균)
            combined_score = 0.5 * entropy_diff + 0.5 * jsd_value
            
            variation_scores.append({
                'entropy_diff': entropy_diff,
                'jsd': jsd_value,
                'combined': combined_score
            })
            
        return variation_scores
    
    def detect_boundaries(self, variation_scores: List[Dict[str, float]]) -> List[int]:
        """
        Step 4: 경계 탐지
        Boundary(t) = 1[V_t ≥ τ]
        
        Args:
            variation_scores: 변화 점수 리스트
            
        Returns:
            boundaries: 경계 인덱스 리스트
        """
        boundaries = []
        
        for i, scores in enumerate(variation_scores):
            # 두 메트릭 중 하나라도 임계값을 넘으면 경계로 판단
            if (scores['entropy_diff'] >= self.entropy_threshold or 
                scores['jsd'] >= self.jsd_threshold):
                boundaries.append(i)
                
        return boundaries
    
    def merge_boundaries(self, boundaries: List[int], n_segments: int) -> List[Tuple[int, int]]:
        """
        Step 5: 경계 병합 및 평활화
        
        Args:
            boundaries: 경계 인덱스 리스트
            n_segments: 전체 세그먼트 수
            
        Returns:
            merged_segments: (시작, 끝) 인덱스 튜플 리스트
        """
        if not boundaries:
            return [(0, n_segments - 1)]
            
        # 시작점 추가
        if 0 not in boundaries:
            boundaries = [0] + boundaries
            
        # 끝점 추가
        if n_segments not in boundaries:
            boundaries.append(n_segments)
            
        # 인접한 경계 병합
        merged_boundaries = [boundaries[0]]
        
        for boundary in boundaries[1:]:
            # 이전 경계와의 거리 확인
            if boundary - merged_boundaries[-1] >= self.min_segment_length:
                merged_boundaries.append(boundary)
                
        # 세그먼트 구간 생성
        segments = []
        for i in range(len(merged_boundaries) - 1):
            segments.append((merged_boundaries[i], merged_boundaries[i+1] - 1))
            
        return segments
    
    def segment(self, topic_distributions: np.ndarray, 
                segments_info: List[Dict] = None) -> List[Dict]:
        """
        전체 동적 세그먼테이션 프로세스
        
        Args:
            topic_distributions: 세그먼트별 토픽 분포
            segments_info: 원본 세그먼트 정보 (선택적)
            
        Returns:
            dynamic_segments: 동적으로 분할된 세그먼트 리스트
        """
        # Step 3: 변화 점수 계산
        variation_scores = self.calculate_variation_scores(topic_distributions)
        
        # Step 4: 경계 탐지
        boundaries = self.detect_boundaries(variation_scores)
        
        # Step 5: 경계 병합
        merged_segments = self.merge_boundaries(boundaries, len(topic_distributions))
        
        # 결과 구성
        dynamic_segments = []
        
        for seg_idx, (start, end) in enumerate(merged_segments):
            segment = {
                'segment_id': seg_idx,
                'start_idx': start,
                'end_idx': end,
                'micro_segments': list(range(start, end + 1)),
                'variation_scores': variation_scores[start:end+1] if variation_scores else []
            }
            
            # 원본 세그먼트 정보가 있으면 추가
            if segments_info:
                segment['original_segments'] = segments_info[start:end+1]
                
                # 시간 정보 계산
                if 'start' in segments_info[start]:
                    segment['start_time'] = segments_info[start]['start']
                    
                if 'end' in segments_info[end]:
                    segment['end_time'] = segments_info[end]['end']
                elif 'start' in segments_info[end] and 'segments' in segments_info[end]:
                    # 마지막 세그먼트의 끝 시간 계산
                    last_seg = segments_info[end]['segments'][-1] if segments_info[end]['segments'] else {}
                    if 'start' in last_seg and 'duration' in last_seg:
                        segment['end_time'] = last_seg['start'] + last_seg['duration']
                        
                # 텍스트 병합
                texts = [seg['text'] for seg in segments_info[start:end+1] 
                        if 'text' in seg]
                if texts:
                    segment['text'] = ' '.join(texts)
                    
                # 토픽 정보 병합
                all_topics = []
                for seg in segments_info[start:end+1]:
                    if 'topics' in seg:
                        all_topics.extend(seg['topics'])
                        
                # 중복 제거 및 평균 확률 계산
                topic_map = {}
                for topic in all_topics:
                    tid = topic['topic_id']
                    if tid in topic_map:
                        topic_map[tid]['probability'] += topic['probability']
                        topic_map[tid]['count'] += 1
                    else:
                        topic_map[tid] = {
                            'topic_id': tid,
                            'probability': topic['probability'],
                            'keywords': topic.get('keywords', []),
                            'count': 1
                        }
                        
                # 평균 계산 및 정렬
                merged_topics = []
                for tid, tinfo in topic_map.items():
                    avg_prob = tinfo['probability'] / tinfo['count']
                    merged_topics.append({
                        'topic_id': tid,
                        'probability': avg_prob,
                        'keywords': tinfo['keywords']
                    })
                    
                merged_topics.sort(key=lambda x: x['probability'], reverse=True)
                segment['topics'] = merged_topics[:5]  # 상위 5개 토픽
                
            dynamic_segments.append(segment)
            
        return dynamic_segments
    
    def evaluate_segmentation(self, segments: List[Dict], 
                            topic_distributions: np.ndarray) -> Dict[str, float]:
        """
        세그먼테이션 품질 평가
        
        Args:
            segments: 세그먼트 리스트
            topic_distributions: 토픽 분포
            
        Returns:
            metrics: 평가 메트릭
        """
        metrics = {
            'num_segments': len(segments),
            'avg_segment_length': 0,
            'intra_segment_similarity': 0,
            'inter_segment_distance': 0
        }
        
        if not segments:
            return metrics
            
        # 평균 세그먼트 길이
        total_length = sum(seg['end_idx'] - seg['start_idx'] + 1 for seg in segments)
        metrics['avg_segment_length'] = total_length / len(segments)
        
        # 세그먼트 내 유사도 (낮을수록 좋음)
        intra_similarities = []
        for seg in segments:
            start, end = seg['start_idx'], seg['end_idx']
            if end > start:
                seg_dists = topic_distributions[start:end+1]
                # 세그먼트 내 JSD 평균
                jsds = []
                for i in range(len(seg_dists)-1):
                    jsd = jensenshannon(seg_dists[i], seg_dists[i+1]) ** 2
                    jsds.append(jsd)
                if jsds:
                    intra_similarities.append(np.mean(jsds))
                    
        if intra_similarities:
            metrics['intra_segment_similarity'] = np.mean(intra_similarities)
            
        # 세그먼트 간 거리 (높을수록 좋음)
        if len(segments) > 1:
            inter_distances = []
            for i in range(len(segments)-1):
                # 각 세그먼트의 평균 분포
                seg1_dist = np.mean(topic_distributions[segments[i]['start_idx']:segments[i]['end_idx']+1], axis=0)
                seg2_dist = np.mean(topic_distributions[segments[i+1]['start_idx']:segments[i+1]['end_idx']+1], axis=0)
                
                # JSD 계산
                jsd = jensenshannon(seg1_dist, seg2_dist) ** 2
                inter_distances.append(jsd)
                
            metrics['inter_segment_distance'] = np.mean(inter_distances)
            
        return metrics 