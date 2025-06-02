"""
YouTube 강의 동적 Segmentation 사용 예제
"""
from main import YouTubeLectureSegmenter
import json

def example_basic():
    """기본 사용 예제"""
    # YouTube URL (예시)
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # 기본 설정으로 실행
    segmenter = YouTubeLectureSegmenter()
    results = segmenter.process_video(video_url)
    
    print("처리 완료!")
    print(f"동적 세그먼트 수: {results['num_dynamic_segments']}")


def example_custom_config():
    """사용자 정의 설정 예제"""
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # 사용자 정의 설정
    config = {
        'language': 'ko',           # 한국어 처리
        'window_size': 30,          # 30초 단위 마이크로 세그먼트
        'n_topics': 7,              # 7개 토픽
        'entropy_threshold': 0.4,   # 더 민감한 엔트로피 임계값
        'jsd_threshold': 0.25,      # 더 민감한 JSD 임계값
        'min_segment_length': 2     # 최소 2개 마이크로 세그먼트
    }
    
    segmenter = YouTubeLectureSegmenter(config)
    results = segmenter.process_video(video_url, output_dir='data/custom_output')
    
    # 결과 분석
    if 'segments' in results:
        print("\n세그먼트별 주요 토픽:")
        for seg in results['segments']:
            print(f"\n세그먼트 {seg['segment_id']+1}:")
            if 'topics' in seg:
                for topic in seg['topics'][:2]:
                    print(f"  - 토픽 {topic['topic_id']}: {topic['probability']:.3f}")


def example_batch_processing():
    """여러 비디오 일괄 처리 예제"""
    video_urls = [
        "https://www.youtube.com/watch?v=video1",
        "https://www.youtube.com/watch?v=video2",
        "https://www.youtube.com/watch?v=video3"
    ]
    
    segmenter = YouTubeLectureSegmenter()
    all_results = []
    
    for url in video_urls:
        try:
            results = segmenter.process_video(url)
            all_results.append(results)
            print(f"✓ 처리 완료: {results['video_id']}")
        except Exception as e:
            print(f"✗ 처리 실패: {url} - {str(e)}")
    
    # 전체 결과 저장
    with open('data/batch_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


def example_result_analysis():
    """결과 분석 예제"""
    # 저장된 결과 파일 읽기
    with open('data/output/VIDEO_ID_segments.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("결과 분석:")
    print(f"처리 시간: {results['processed_at']}")
    print(f"총 세그먼트 수: {results['num_dynamic_segments']}")
    
    # 세그먼트 길이 분포
    segment_lengths = []
    for seg in results['segments']:
        length = len(seg['micro_segments'])
        segment_lengths.append(length)
    
    print(f"\n세그먼트 길이 통계:")
    print(f"  - 최소: {min(segment_lengths)}")
    print(f"  - 최대: {max(segment_lengths)}")
    print(f"  - 평균: {sum(segment_lengths)/len(segment_lengths):.2f}")
    
    # 토픽 분포
    topic_counts = {}
    for seg in results['segments']:
        if 'topics' in seg:
            for topic in seg['topics']:
                tid = topic['topic_id']
                topic_counts[tid] = topic_counts.get(tid, 0) + 1
    
    print(f"\n토픽 출현 빈도:")
    for tid, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - 토픽 {tid}: {count}회")


if __name__ == '__main__':
    print("YouTube 강의 동적 Segmentation 예제")
    print("="*50)
    
    # 기본 예제 실행
    print("\n1. 기본 사용 예제")
    try:
        example_basic()
    except Exception as e:
        print(f"예제 실행 중 오류: {str(e)}")
    
    # 다른 예제들은 필요에 따라 주석 해제하여 실행
    # example_custom_config()
    # example_batch_processing()
    # example_result_analysis() 