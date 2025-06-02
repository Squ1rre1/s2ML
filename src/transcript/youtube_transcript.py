"""
YouTube 동영상 자막 추출 모듈
"""
import re
from typing import List, Dict, Optional, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import logging

logger = logging.getLogger(__name__)


class YouTubeTranscriptExtractor:
    """YouTube 동영상에서 자막을 추출하는 클래스"""
    
    def __init__(self):
        self.supported_languages = ['ko', 'en']  # 지원 언어 우선순위
        
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        YouTube URL에서 video ID를 추출
        
        Args:
            url: YouTube 동영상 URL
            
        Returns:
            video_id: 추출된 video ID
        """
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',
            r'youtube\.com\/embed\/([^&\n?]+)',
            r'youtube\.com\/v\/([^&\n?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # URL이 아닌 경우 직접 video_id일 수 있음
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url
            
        return None
    
    def get_transcript(self, video_id: str, language: Optional[str] = None) -> List[Dict]:
        """
        비디오 ID로 자막 데이터 가져오기
        
        Args:
            video_id: YouTube 비디오 ID
            language: 선호 언어 코드 (기본값: None - 자동 선택)
            
        Returns:
            transcript: 자막 데이터 리스트
                - text: 자막 텍스트
                - start: 시작 시간 (초)
                - duration: 지속 시간 (초)
        """
        try:
            if language:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            else:
                # 사용 가능한 자막 목록 확인
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # 수동 자막 우선, 그 다음 자동 생성 자막
                transcript = None
                for lang in self.supported_languages:
                    try:
                        transcript = transcript_list.find_transcript([lang]).fetch()
                        logger.info(f"Found transcript in language: {lang}")
                        break
                    except:
                        continue
                
                # 지원 언어가 없으면 첫 번째 사용 가능한 자막 사용
                if not transcript:
                    transcript = next(iter(transcript_list)).fetch()
                    
            return transcript
            
        except TranscriptsDisabled:
            logger.error(f"Transcripts are disabled for video: {video_id}")
            raise
        except NoTranscriptFound:
            logger.error(f"No transcript found for video: {video_id}")
            raise
        except Exception as e:
            logger.error(f"Error fetching transcript: {str(e)}")
            raise
    
    def process_transcript(self, transcript: List[Dict], window_size: int = 60) -> List[Dict]:
        """
        자막을 시간 윈도우 단위로 그룹화
        
        Args:
            transcript: 원본 자막 데이터
            window_size: 윈도우 크기 (초)
            
        Returns:
            processed_transcript: 그룹화된 자막 데이터
        """
        if not transcript:
            return []
            
        processed = []
        current_window = {
            'text': '',
            'start': transcript[0]['start'],
            'end': transcript[0]['start'] + window_size,
            'segments': []
        }
        
        for segment in transcript:
            segment_end = segment['start'] + segment['duration']
            
            # 현재 윈도우에 포함되는 경우
            if segment['start'] < current_window['end']:
                current_window['text'] += ' ' + segment['text']
                current_window['segments'].append({
                    'text': segment['text'],
                    'start': segment['start'],
                    'duration': segment['duration']
                })
            else:
                # 새로운 윈도우 시작
                if current_window['text'].strip():
                    processed.append(current_window)
                
                current_window = {
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['start'] + window_size,
                    'segments': [{
                        'text': segment['text'],
                        'start': segment['start'],
                        'duration': segment['duration']
                    }]
                }
        
        # 마지막 윈도우 추가
        if current_window['text'].strip():
            processed.append(current_window)
            
        return processed
    
    def extract_from_url(self, url: str, window_size: int = 60) -> Tuple[str, List[Dict]]:
        """
        URL에서 자막을 추출하고 처리
        
        Args:
            url: YouTube 동영상 URL
            window_size: 처리할 윈도우 크기 (초)
            
        Returns:
            video_id: 비디오 ID
            processed_transcript: 처리된 자막 데이터
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {url}")
            
        transcript = self.get_transcript(video_id)
        processed = self.process_transcript(transcript, window_size)
        
        return video_id, processed 