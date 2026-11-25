# 실행 확인용 합성 데이터

`synthetic/`의 파일은 코드 실행을 확인하기 위해 생성한 인공 데이터다.
BRCA1 서열, Findlay 실험 점수, ESM-2 임베딩, AlphaFold 예측 결과가 아니다.
여기서 나온 성능 수치를 실제 단백질 예측 성능으로 해석하면 안 된다.

같은 폴더에 있는 `BRCA1_HUMAN_Findlay_2018.csv`, `brca1_esm2_features.parquet`,
`AF-P38398-F1-confidence_v6.json`은 현재 원본 데이터의 출력 요약을 담은 파일이므로
실험 입력으로 사용하지 않는다. 아래 명령은 새로 만든 `synthetic/` 파일을 사용한다.

## 파일 구성

| 파일 | 내용 |
| --- | --- |
| `synthetic/variants.csv` | 학습 변이 192개. 전체 변이 서열, 인공 DMS 점수와 0/1 라벨 |
| `synthetic/esm_features.parquet` | 192행 × 1,287열. 임베딩 모양의 난수 1,280개, 코사인 유사도, 식별자와 정답 |
| `synthetic/confidence.json` | 80개 잔기의 인공 pLDDT. AlphaFold confidence JSON과 같은 필드 구조 |
| `synthetic/wt.fasta` | 길이 80의 인공 WT 서열 |
| `synthetic/prediction_variants.csv` | 정답이 없는 예측 대상 변이 48개와 전체 서열 |
| `synthetic/prediction_features.parquet` | 예측용 48행 × 1,285열. 정답 열 없음 |
| `synthetic/manifest.json` | 생성 seed, 행 수, 출처가 인공 데이터라는 설명 |
| `generate_synthetic.py` | 위 데이터 생성 코드. 이미 있는 폴더는 덮어쓰지 않음 |

각 위치에 네 개의 치환이 있다. 학습 위치는 1~48, 예측 위치는 49~60으로 서로 겹치지 않는다.
정답은 전하/소수성 등으로 만든 인위적인 규칙에 잡음을 더한 값이다.
임베딩은 고정 seed로 만든 난수 벡터다. ESM 모델을 다운로드하거나 실행하지 않고 실험할 수 있다.

## 실행

프로젝트의 `feature-lab` 폴더에서 필요한 패키지를 설치한다.

```powershell
python -m pip install -r requirements.txt
```

이 예제에는 CIF가 없으므로 전하와 소수성 후보를 회귀로 비교한다.
기본 ESM + 개별 후보 2개 + 전체 조합, 총 4개 조합을 같은 3-fold로 평가한다.

```powershell
python experiment.py compare --input examples/synthetic/esm_features.parquet --features charge hydropathy --task regression --folds 3 --out examples/synthetic/results_compare
```

분류 모델 하나를 학습하고 교차검증한다.

```powershell
python experiment.py train --input examples/synthetic/esm_features.parquet --features esm charge hydropathy --task classification --folds 3 --out examples/synthetic/results_classification
```

회귀 비교에서 저장한 전체 feature 조합 모델로 새 변이 48개의 점수를 예측한다.

```powershell
python experiment.py predict --input examples/synthetic/prediction_features.parquet --model examples/synthetic/results_compare/esm+charge+hydropathy/model.joblib --out examples/synthetic/results_prediction/predictions.csv
```

분류 모델로 확률을 예측하려면 위 명령의 `--model`을
`examples/synthetic/results_classification/esm+charge+hydropathy/model.joblib`로,
`--out`을 `examples/synthetic/results_prediction/classification_probabilities.csv`로 바꾼다.

이미 생성된 출력 경로는 덮어쓰지 않는다. 반복 실험은 `--out`에 새 경로를 지정한다.
`--folds 5`도 사용할 수 있다. 분류 라벨은 전체 데이터뿐 아니라 각 학습 fold에도 두 종류가 있어야 한다.

실제 ESM 추출기를 시험하고 싶다면 `synthetic/variants.csv`를 입력할 수 있지만,
별도로 `requirements-esm.txt` 설치 및 ESM 가중치 로드가 필요하다.
일반적인 train/compare/predict 확인에는 포함된 Parquet를 바로 사용하면 된다.

새 `structure` 후보는 SASA·주변 잔기 수·pLDDT의 세 열을 묶는다. 이 예제의
confidence JSON에는 좌표가 없으므로 구조 그룹 전체를 계산할 수 없다.
구조 그룹 실험에는 같은 참조 서열의 WT CIF 또는 미리 계산한 구조 표가 필요하다.
JSON을 이용한 기존 pLDDT 단독 실험은 `train --features esm plddt --structure ...`로 가능하다.
