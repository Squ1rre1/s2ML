# YouTube 강의 동적 Segmentation 프로젝트

## 프로젝트 개요
YouTube 강의 영상의 자막을 분석하여 토픽 기반으로 동적 segmentation을 수행하는 프로젝트입니다. 기존의 고정 시간(5분) 단위 분할 방식 대신, 토픽의 변화와 coverage를 기반으로 의미 있는 구간을 자동으로 분할합니다.

## 주요 기능
1. **자막 추출**: YouTube Transcript API를 통한 자막 데이터 수집
2. **토픽 모델링**: Wikification을 통한 개념 추출 및 토픽 모델링
3. **동적 Segmentation**: 엔트로피와 JSD(Jensen-Shannon Divergence) 기반 경계 탐지
4. **Top-k 키워드 추출**: 각 세그먼트별 핵심 키워드 추출

## 시스템 아키텍처
```
YouTube Video → Transcript API → Wikification → Topic Modeling → Dynamic Segmentation → Segmented Lectures
```

## 알고리즘 개요
1. **정규화**: 각 micro-segment의 토픽 분포 정규화
2. **엔트로피 계산**: 토픽 분포의 엔트로피 계산
3. **변화점 탐지**: 엔트로피 변화량 및 JSD를 통한 경계 후보 탐지
4. **경계 결정**: 임계값 기반 경계 결정
5. **경계 평활화**: 인접 경계 병합 및 노이즈 제거

## 프로젝트 구조
```
s2ML/
├── src/
│   ├── transcript/      # YouTube 자막 추출
│   ├── wikification/    # Wikification 및 토픽 추출
│   ├── segmentation/    # 동적 segmentation 알고리즘
│   └── utils/          # 유틸리티 함수
├── data/               # 데이터 저장
├── tests/              # 테스트 코드
├── requirements.txt    # 의존성 패키지
└── README.md
```