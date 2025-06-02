"""
YouTube 강의 동적 Segmentation 메인 실행 파일
"""
import os
import json
import logging
from typing import Dict, List
import argparse
from datetime import datetime

from src.transcript.youtube_transcript import YouTubeTranscriptExtractor
from src.wikification.wikifier import Wikifier
from src.segmentation.dynamic_segmenter import DynamicSegmenter

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YouTubeLectureSegmenter:
    """YouTube 강의 동적 세그먼테이션 파이프라인"""
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        
        # 컴포넌트 초기화
        self.transcript_extractor = YouTubeTranscriptExtractor()
        self.wikifier = Wikifier(language=self.config.get('language', 'ko'))
        self.segmenter = DynamicSegmenter(
            entropy_threshold=self.config.get('entropy_threshold', 0.5),
            jsd_threshold=self.config.get('jsd_threshold', 0.3),
            min_segment_length=self.config.get('min_segment_length', 3)
        )
        
    def process_video(self, video_url: str, output_dir: str = 'data/output') -> Dict:
        """
        비디오 처리 파이프라인 실행
        
        Args:
            video_url: YouTube 비디오 URL
            output_dir: 출력 디렉토리
            
        Returns:
            results: 처리 결과
        """
        logger.info(f"Processing video: {video_url}")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'video_url': video_url,
            'processed_at': datetime.now().isoformat(),
            'config': self.config
        }
        
        try:
            # 1. 자막 추출
            logger.info("Extracting transcript...")
            video_id, transcript_segments = self.transcript_extractor.extract_from_url(
                video_url, 
                window_size=self.config.get('window_size', 60)
            )
            results['video_id'] = video_id
            results['num_micro_segments'] = len(transcript_segments)
            
            logger.info(f"Extracted {len(transcript_segments)} micro-segments")
            
            # 2. Wikification 및 토픽 모델링
            logger.info("Performing wikification and topic modeling...")
            processed_segments = self.wikifier.process_segments(
                transcript_segments,
                n_topics=self.config.get('n_topics', 5)
            )
            
            # 토픽 분포 계산
            topic_distributions = self.wikifier.calculate_topic_distribution(processed_segments)
            
            logger.info(f"Identified {topic_distributions.shape[1]} topics")
            
            # 3. 동적 세그먼테이션
            logger.info("Performing dynamic segmentation...")
            dynamic_segments = self.segmenter.segment(
                topic_distributions,
                processed_segments
            )
            
            results['num_dynamic_segments'] = len(dynamic_segments)
            logger.info(f"Created {len(dynamic_segments)} dynamic segments")
            
            # 4. 평가 메트릭 계산
            metrics = self.segmenter.evaluate_segmentation(
                dynamic_segments,
                topic_distributions
            )
            results['metrics'] = metrics
            
            # 5. 결과 저장
            results['segments'] = dynamic_segments
            
            # 전체 결과 저장
            output_path = os.path.join(output_dir, f"{video_id}_segments.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Results saved to: {output_path}")
            
            # 요약 정보 출력
            self._print_summary(results)
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def _print_summary(self, results: Dict):
        """처리 결과 요약 출력"""
        print("\n" + "="*60)
        print("처리 결과 요약")
        print("="*60)
        print(f"비디오 ID: {results.get('video_id', 'N/A')}")
        print(f"원본 세그먼트 수: {results.get('num_micro_segments', 0)}")
        print(f"동적 세그먼트 수: {results.get('num_dynamic_segments', 0)}")
        
        if 'metrics' in results:
            print(f"\n평가 메트릭:")
            print(f"  - 평균 세그먼트 길이: {results['metrics']['avg_segment_length']:.2f}")
            print(f"  - 세그먼트 내 유사도: {results['metrics']['intra_segment_similarity']:.4f}")
            print(f"  - 세그먼트 간 거리: {results['metrics']['inter_segment_distance']:.4f}")
            
        if 'segments' in results:
            print(f"\n세그먼트 정보:")
            for seg in results['segments']:
                start_time = seg.get('start_time', 0)
                end_time = seg.get('end_time', 0)
                duration = end_time - start_time
                
                print(f"\n  세그먼트 {seg['segment_id']+1}:")
                print(f"    시간: {self._format_time(start_time)} - {self._format_time(end_time)} ({duration:.1f}초)")
                print(f"    마이크로 세그먼트: {len(seg['micro_segments'])}개")
                
                if 'topics' in seg and seg['topics']:
                    print(f"    주요 토픽:")
                    for topic in seg['topics'][:3]:
                        keywords = ', '.join([kw[0] for kw in topic['keywords'][:3]])
                        print(f"      - 토픽 {topic['topic_id']}: {keywords} (확률: {topic['probability']:.3f})")
                        
        print("="*60 + "\n")
        
    def _format_time(self, seconds: float) -> str:
        """초를 MM:SS 형식으로 변환"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='YouTube 강의 동적 Segmentation')
    parser.add_argument('url', type=str, help='YouTube 비디오 URL')
    parser.add_argument('--language', type=str, default='ko', choices=['ko', 'en'],
                       help='처리 언어 (기본값: ko)')
    parser.add_argument('--window-size', type=int, default=60,
                       help='마이크로 세그먼트 윈도우 크기 (초, 기본값: 60)')
    parser.add_argument('--n-topics', type=int, default=5,
                       help='토픽 수 (기본값: 5)')
    parser.add_argument('--entropy-threshold', type=float, default=0.5,
                       help='엔트로피 변화 임계값 (기본값: 0.5)')
    parser.add_argument('--jsd-threshold', type=float, default=0.3,
                       help='JSD 임계값 (기본값: 0.3)')
    parser.add_argument('--min-segment-length', type=int, default=3,
                       help='최소 세그먼트 길이 (기본값: 3)')
    parser.add_argument('--output-dir', type=str, default='data/output',
                       help='출력 디렉토리 (기본값: data/output)')
    
    args = parser.parse_args()
    
    # 설정 생성
    config = {
        'language': args.language,
        'window_size': args.window_size,
        'n_topics': args.n_topics,
        'entropy_threshold': args.entropy_threshold,
        'jsd_threshold': args.jsd_threshold,
        'min_segment_length': args.min_segment_length
    }
    
    # 파이프라인 실행
    segmenter = YouTubeLectureSegmenter(config)
    results = segmenter.process_video(args.url, args.output_dir)
    
    return results


if __name__ == '__main__':
    main() 