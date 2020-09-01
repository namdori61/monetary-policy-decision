# monetary-policy-decision
Predict monetary policy voting result with monetary policy decision text using PLM(Pre-trained Language Model): KB-ALBERT-KO

## Acknowledgement
- [KB-ALBERT를 활용한 금융 자연어 서비스 아이디어](http://www.kbdatory.com/competitions/list): 태스크의 목표인 Competition
- [KB-ALBERT-KO](https://github.com/KB-Bank-AI/KB-ALBERT-KO): 위의 Competition과 관련하여 KB금융에서 제공해주신 금융 특화 ALBERT 모델 (public)
- [pyhwp](https://pypi.org/project/pyhwp/): hwp 파일을 txt 파일로 변환

## References
- 김미숙. Prediction of stock price, base rate, and interest rate spread with text data. Diss. 서울대학교 대학원, 2018.
- Kim, Soohyon, et al. "경제 분석을 위한 텍스트 마이닝 (Text Mining for Economic Analysis)." Available at SSRN 3405781 (2019).

## Goal
- 금통위 의결문을 이용한 위원 투표 결과 분류 (소수 의견 탐지)

## Background
금융 도메인 기반
- 금융 통화 위원회의 통화 정책의 시중 금리에 대한 영향력 
- 통화 정책에 의한 시중 금리 변화에는 금융 통화 위원회 회의 결과 중 소수의견 중요

머신러닝 기반
- 금융 통화 위원회 회의는 연 8회 (2017년 이후), 연 12회 (2017년 이전) 발생하는 이벤트로 그 샘플이 매우 적어서 머신러닝 활용이 어려움
- 위의 데이터 부족 문제에 대한 솔루션으로서 금융 도메인 텍스트 기반 PLM(Pre-trained Language Model)의 유용성 

## Data
- 한국은행 금융통화 위원회 의결사항 수집 : https://www.bok.or.kr/portal/bbs/P0000093/list.do?menuNo=200789
- 2006년 이전 데이터는 모두 hwp 형식 임을 감안하여 모든 데이터를 hwp 파일로 수집 후 txt로 변환
- 실행 방법
  - scrap: `python scrap_data.py --save_dir [DATA_SAVE_DIR] --page_num [PAGE_NUM]`
  - hwp to txt (pyhwp 패키지 이용 및 변환 오류 건에 대해서는 직접 수행): `python transform_hwp_txt.py --dir [DATA_SAVE_DIR]`
- 데이터 저작권 및 이용 문의 완료 (출처 공개 하에 연구/영리 목적 제한 없)

## Preprocess
1. 한문 번역
2. 불필요 특수 문자 및 공간 제거

## Model
1. KB-ALBERT-KO pulbic
2. KB-ALBERT-KO private (KB금융 측에 별도로 신청한 모델)
3. mBERT
4. KoBERT
5. KcBERT

## Train

## Evaluate