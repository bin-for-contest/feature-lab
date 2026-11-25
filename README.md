# BRCA1 feature 실험

ESM-2 feature에 추가 특성을 결합하고 LightGBM으로 Findlay DMS 기능 점수를 예측한다.
동일 위치의 변이는 같은 fold에 배정하며, 모든 비교 조합은 같은 fold를 사용한다.

## 추가 feature 후보

| CLI 이름 | 값과 출처 |
| --- | --- |
| `charge` | PyPI `peptides.Peptide.charge(pH=7.0, pKscale="Lehninger")`의 Mut − WT 차이 |
| `hydropathy` | `peptides.Peptide.hydrophobicity(scale="KyteDoolittle")`의 Mut − WT 차이 |
| `structure` | WT CIF에서 계산한 SASA, Cα 기준 8 Å 이내 주변 잔기 수, pLDDT를 묶은 3개 열 |
| `hbond` | WT CIF에서 변이 위치의 곁사슬이 다른 잔기에 수소결합을 제공하거나 받을 수 있는 원자쌍 수, 역할별 2개 열 |

전하와 소수성은 서로 다른 특성이므로 별도 후보로 비교한다. `peptides`는 단일 잔기에
적용하며, 값은 전체 단백질의 평균 소수성이나 총전하가 아닌 치환 잔기의 차이다.
`polarity`와 BLOSUM은 추가 후보에 포함하지 않는다.

`peptides`는 서열 descriptor 계산 패키지다. AlphaFold 구조 파일, 실제 구조의 수소결합,
Findlay 실험 정답 데이터는 제공하지 않는다. `hbond` 근사치는 이 패키지에서 온 값이 아니다.
pLDDT는 구조 좌표 자체를 표현하는 feature가 아니라 구조 예측의 신뢰도다.

## 실행

```powershell
python -m pip install -r requirements.txt
python experiment.py compare --input brca1_esm2_features.parquet --features charge hydropathy --out results
```

ESM-2 추출까지 실행하려면 `requirements-esm.txt`를 설치한다. 구조 후보를 비교하려면
`--features`에 `structure`를 추가하고 `--structure`로 AlphaFold WT CIF/mmCIF를 지정한다.
SASA·주변 잔기 수·pLDDT를 각각 별도 열로 유지하면서 하나의 후보 그룹으로 비교한다.
confidence JSON만으로는 좌표가 없어 구조 그룹 전체를 만들 수 없다.
기본 `esm`에는 임베딩 1,280개와 코사인 유사도, 총 1,281개 feature가 포함된다.
`esm_llr`은 추출·학습에서 제외한다. 기존 입력 파일에 이 열이 남아 있어도 사용하지 않는다.
비교는 기본 ESM, 각 후보를 하나씩 추가한 조합, 선택한 후보를 모두 추가한 조합을 실행한다.
LLR을 포함해 학습한 기존 ESM 모델은 새 feature 구성으로 다시 학습해야 한다.

전하 계산을 기존 정수 근사에서 pH 기반 계산으로 바꿨으므로 `charge`를 사용한 기존 모델은
다시 학습해야 한다. 새 열 이름은 `delta_charge_ph7_lehninger`이며 기존 열 이름과 다르면
예측 단계에서 오류로 중단한다.

## 전체 코드 구조

```text
feature-lab/                         현재 실험을 실행하는 프로젝트
    extract_esm2_embeddings.py         서열 검증, ESM-2 feature 추출 CLI
    features.py                       표 읽기, 변이/구조 검증, feature 행렬 생성
    parse_cif.py                      mmCIF 전체 사전 및 원자별 표 파싱
    structure_features.py             WT CIF에서 위치별 구조 feature 계산
    experiment.py                     LightGBM 학습, 조합 비교, 저장 모델 예측 CLI
    requirements.txt                  학습/비교에 필요한 패키지
    requirements-esm.txt              기본 패키지 + ESM-2 추출 패키지
    examples/toy.csv                   입출력 흐름 확인용 작은 예제
```

실행은 `feature-lab` 폴더를 기준으로 한다. 현재 학습 코드는 원본 스크립트를 import하지 않는다.
원본은 비교용이며, 새 작업에서는 `feature-lab`의 추출기를 사용한다.
원본 스크립트는 main 가드가 없어 import만 해도 모델 실행이 시작된다.

전체 처리는 다음 순서로 나뉜다.

1. 변이 식별자와 서열을 검증한다.
2. 고정된 ESM-2로 변이 위치의 벡터와 코사인 유사도를 추출해 파일로 저장한다.
3. 저장된 표에 선택한 잔기 속성 차이와 선택적 구조 feature를 결합한다.
4. 위치별 교차검증으로 LightGBM의 OOF 예측과 성능을 계산한다.
5. 전체 데이터로 최종 모델을 다시 학습해 저장한다.
6. 새 변이에는 같은 규칙으로 feature를 만들고 저장 모델로 예측한다.

ESM 추출은 비용이 큰 단계이므로 한 번 만든 표를 여러 feature 조합 실험에서 재사용한다.
ESM-2와 LightGBM을 하나의 신경망으로 연결해 동시에 학습하는 구조는 아니다.

## 입력 데이터와 행의 의미

한 행은 한 단백질의 단일 아미노산 치환이다. `C39Y`는 전체 참조 서열의 39번 C를
Y로 바꾼 변이다. 여러 단백질이나 isoform을 같은 표에 섞는 것을 전제로 하지 않는다.

| 열 | 의미 | 필요한 시점 |
| --- | --- | --- |
| `mutant` | `C39Y` 같은 단일 치환 식별자 | 모든 단계 |
| `wt`, `pos`, `mut` | WT(wildtype) 잔기, 전체 서열의 1-based 위치, Mut 잔기 | `mutant`에서 자동 복원. 이미 있으면 일치 검사 |
| `mutated_sequence` | 치환을 반영한 전체 단백질 서열 | ESM 추출. 없으면 `--reference-fasta`로 생성 |
| `DMS_score` | 기능 점수 정답 | 회귀 학습/평가 |
| `DMS_score_bin` | 입력에 제공된 0/1 정답 | 분류 학습/평가 |
| `esm_emb_0` ~ `esm_emb_1279` | 변이 위치의 ESM 벡터 성분 | ESM feature를 선택한 학습/예측 |
| `esm_cos_sim` | WT와 Mut 잔기 벡터의 방향 유사도 | `esm` 또는 `cosine` 그룹 선택 시 |
| `sasa` | WT 잔기의 용매 접근 표면적(Å²) | `structure` 선택 시 |
| `neighbor_count` | Cα 기준 8 Å 이내 다른 WT 잔기 수 | `structure` 선택 시 |
| `plddt` | 위치별 WT AlphaFold 예측 신뢰도 | `structure` 또는 호환용 `plddt` 선택 시 |

추론에는 정답이 필요 없다. 다만 새 서열의 ESM feature는 먼저 추출해야 한다.
추출 결과에는 `mutated_sequence`를 남기지 않고 식별자, 선택적 정답, 수치 feature만 저장한다.
정답 열과 `pos`는 모델 입력으로 자동 선택하지 않는다. `pos`는 구조 매핑과 CV 그룹에 사용한다.

## extract_esm2_embeddings.py: 서열을 수치 feature로 바꾸는 단계

| 함수 | 역할 |
| --- | --- |
| `get_window_seq()` | 전체 위치를 기준으로 서열을 자르고 window 내부 위치 반환 |
| `reference_sequence()` | 첫 변이에서 WT를 복원하고 모든 행이 그 WT의 단일 치환인지 검사 |
| `with_fasta()` | 단일 WT FASTA에서 각 변이의 전체 서열을 생성하고 기존 서열과 대조 |
| `extract()` | WT 위치별 캐시와 변이별 임베딩을 계산해 출력 표 생성 |
| `main()` | 옵션 처리, 검증, 추출, CSV/Parquet 저장 |

window는 변이 위치를 중심으로 양옆 510개씩, 최대 1,021개 잔기를 사용한다.
예를 들어 위치 1에서는 앞쪽 잔기가 없으므로 최대 511개 잔기만 사용한다.
이는 전체 서열을 일정 stride로 훑어 평균내는 방식과 다르다.

`pos`는 1부터 시작하지만 Python 문자열 인덱스는 0부터 시작한다.
window를 잘라낸 뒤에는 `offset = pos - 1 - window_start`로 상대 위치를 구한다.
ESM 입력의 맨 앞에는 `<cls>`가 붙으므로 출력 벡터를 고를 때는 다시 1을 더한다.
배치마다 window 길이가 다를 수 있어 padding이 붙지만, 선택하는 위치는 실제 잔기다.

ESM-2 650M의 33번째 레이어 출력에서 변이 잔기의 1,280차원 벡터 하나를 선택한다.
배치 출력이 `(B, 토큰 수, 1280)`이면 잔기를 고른 결과는 `(B, 1280)`이다.
WT 벡터는 위치별로 한 번만 구해 캐싱하고, Mut 벡터는 변이마다 계산한다.
둘의 코사인 유사도는 다음과 같다.

```text
esm_cos_sim = dot(WT_vector, Mut_vector)
              / (norm(WT_vector) * norm(Mut_vector) + 1e-8)
```

`model.eval()`과 `torch.inference_mode()`를 사용하며 ESM 가중치를 업데이트하지 않는다.
`--batch-size`는 동시에 처리하는 window 수다. 기본값은 2이고 `--device auto`는
CUDA가 가능하면 GPU를, 그렇지 않으면 CPU를 선택한다.
처음 사용하는 가중치가 캐시에 없으면 다운로드가 필요할 수 있다.

## features.py: 검증과 feature 조립

| 함수 | 역할 |
| --- | --- |
| `read_table()` | 확장자에 따라 CSV/TSV/Parquet 읽기 |
| `variants()` | 치환 형식, 메타데이터 일치, 중복, 위치별 WT 일관성 검사 |
| `peptide_scale()` | peptides로 20개 잔기의 전하/소수성을 계산하고 캐싱 |
| `read_confidence_json()` | AlphaFold confidence JSON의 두 배열을 `pos,plddt` 표로 변환 |
| `add_structure()` | CIF에서 구조 feature 계산 또는 JSON/PDB/표 읽기 후 변이 위치별 매핑 |
| `design_matrix()` | 지정한 그룹 순서대로 수치 feature를 연결하고 값 검증 |

전하와 소수성은 `property(Mut) - property(WT)`를 사용한다.
예를 들어 A에서 D로 바뀌면 소수성 차이는 `-3.5 - 1.8 = -5.3`이다.
이 값은 단백질 전체의 평균 소수성 변화량이 아니라 치환 잔기 자체의 속성 차이다.
전하도 pH 7, Lehninger 척도에서 단일 잔기에 계산한 값의 차이를 사용한다.

`hbond`는 WT CIF의 첫 모델에서 체인 A의 서로 다른 두 잔기 사이를 검사한다.
일반적인 이온화 상태에서 donor/acceptor가 될 수 있는 N/O 원자쌍 중 거리가 2.4~3.5 Å이고,
각 끝에서 바로 결합된 중원자 방향과 상대 원자 방향이 모두 90° 이상 벌어진 쌍을 센다.
수소 좌표가 없는 구조에 적용하기 위해 정한 기하학적 근사다. [중원자 거리·각도를 사용한 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC2253219/)

변이 위치에서는 곁사슬만 세며 상대 잔기는 주사슬·곁사슬 모두 허용한다.
donor인 후보 수와 acceptor인 후보 수를 따로 기록한다. 히스티딘과 하이드록실처럼
두 역할이 가능한 경우에는 각 방향이 각각 후보가 될 수 있다. 물과 황을 매개로 한 상호작용은 제외한다.
두 열은 WT 환경을 나타내므로 같은 위치의 모든 치환이 같은 값을 받는다.
기존 모델에 `delta_donor`, `delta_acceptor`가 저장돼 있으면 예측 시 기존의 Mut − WT 규칙을 사용한다.

`AF-P38398-F1-confidence_v6.json` 같은 AlphaFold confidence JSON을 직접 입력할 수 있다.
`residueNumber`를 `pos`, `confidenceScore`를 `plddt`로 연결한다. 두 배열은 비어 있지 않고
길이가 같아야 하며 숫자만 허용한다. pLDDT는 0~100의 유한한 값이어야 한다.
배열이 정렬돼 있지 않아도 실제 `residueNumber`를 기준으로 매핑한다.
`confidenceCategory`는 사용하지 않는다. JSON에는 WT 문자가 없으므로 단백질/isoform의
아미노산 일치는 확인할 수 없으며, 실험과 같은 참조 서열의 confidence 파일을 사용해야 한다.

`structure_features.calculate_structure_features()`는 CIF 첫 모델의 label_asym_id='A'에서
표준 아미노산의 비수소 ATOM 좌표를 사용한다. 잔기 번호는 label_seq_id이며 WT 문자도
변이 표와 대조한다. 원자의 기본 위치 또는 A 대체 위치를 선택하며 중복 원자는 거부한다.
잔기마다 Cα가 정확히 하나 있어야 한다.

- `sasa`: 선택한 체인 전체에서 Shrake–Rupley로 계산한 잔기별 면적(Å²).
  탐침 반지름 1.4 Å, 원자 표면 샘플 100개를 사용한다. 물·다른 체인·금속은 포함하지 않는다.
- `neighbor_count`: Cα 사이 거리가 8 Å 이하인 다른 잔기의 수. 자기 자신은 제외하고
  서열상 바로 이웃한 잔기는 포함한다. 원자가 아니라 서로 다른 잔기를 센다.
- `plddt`: 같은 WT 잔기의 Cα에 기록된 B_iso_or_equiv 값.

세 값은 WT 구조의 환경 정보이므로 같은 위치의 여러 치환에는 동일하게 매핑한다.
SASA는 면적이며 RSA로 정규화하지 않는다. 구조가 불확실한 영역의 기하학적 feature는
예측 오차를 포함할 수 있다. 기능 부위 거리나 실제 수소결합 계산은 이 그룹에 포함하지 않는다.
학습/비교 실행 시 한 번 계산해 모든 fold와 조합에서 재사용한다. 추론 시에도 같은 정의를 쓴다.

기존 pLDDT 단독 모델을 위해 `train --features plddt`와 저장 모델의 `predict`는 유지한다.
`compare`의 후보는 `charge`, `hydropathy`, `hbond`, `structure`다.
`structure`와 `plddt`를 동시에 선택하면 같은 열을 중복 사용하므로 거부한다.
미리 계산한 표를 쓸 경우 `pos,sasa,neighbor_count,plddt` 열을 제공하면 되고 `wt`는 선택 사항이다.

PDB를 입력하면 기존 호환 경로에서 첫 모델의 chain A Cα 원자만 읽고, AlphaFold PDB의
B-factor 칸을 pLDDT로 해석한다. 일반 실험 PDB의 B-factor와 같은 의미로 취급하면 안 된다.
잔기 번호와 WT 문자를 검사하며 삽입 코드나 모호한 위치 매핑은 거부한다.
구조 표에는 중복 없는 양의 정수 `pos`와 선택한 그룹의 열을 넣는다. `wt`를 넣으면 변이 표와 대조한다.
최종 feature에 결측/무한값이 있거나 pLDDT가 0~100 범위 밖이면 학습 전에 중단한다.

| feature 그룹 | 차원 | 생성하는 열 |
| --- | ---: | --- |
| `esm` | 1,281 | 임베딩 1,280개 + `esm_cos_sim` |
| `embedding` | 1,280 | 임베딩만 |
| `cosine` | 1 | `esm_cos_sim`만 |
| `charge` | 1 | `delta_charge_ph7_lehninger` |
| `hydropathy` | 1 | `delta_hydropathy` |
| `hbond` | 2 | `wt_sidechain_hbond_donor_count`, `wt_sidechain_hbond_acceptor_count` |
| `structure` | 3 | `sasa`, `neighbor_count`, `plddt` |
| `plddt` | 1 | 기존 모델 호환/단독 train용 `plddt` |

기본 ESM과 추가 후보 네 가지를 모두 쓰면 총 1,288차원이다.
`esm`과 `embedding`/`cosine`을 동시에 선택해 중복 열을 만드는 것은 허용하지 않는다.

## experiment.py: 학습, 비교, 예측

| 함수 | 역할 |
| --- | --- |
| `metrics()` | 회귀/분류 지표 계산 |
| `predict()` | 회귀 점수 또는 클래스 1의 확률 반환 |
| `evaluate()` | 조합 하나의 CV 평가, OOF 저장, 전체 데이터 재학습 및 모델 저장 |
| `run()` | 데이터 검증, 공통 fold 생성, 여러 조합 실행, 최종 결과 폴더 확정 |
| `infer()` | 저장 모델의 feature 규칙과 열 순서를 재사용해 예측 |
| `write_json()` | 설정/열 목록을 JSON으로 저장 |
| `main()` | `train`, `compare`, `predict` 명령 분기 |

`train`은 지정한 조합 하나를 평가하고 최종 모델을 저장한다. ESM 없이 `charge`만 쓰는
실험도 가능하다. `compare`는 ESM을 고정 baseline으로 두고 각 후보를 하나씩 추가한 경우와
전체를 추가한 경우를 비교한다. 후보 네 가지를 지정하면 기본 1개 + 개별 4개 + 전체 1개,
총 6개 조합을 실행한다. 모든 부분집합을 자동으로 탐색하지는 않는다.

회귀는 `LGBMRegressor`, 분류는 `LGBMClassifier`를 사용한다. 전처리 표준화 없이
단계가 하나인 sklearn Pipeline으로 감싼다. `random_state=0`, `verbosity=-1`을 지정하고
그 외 하이퍼파라미터는 LightGBM 기본값을 사용한다. 현재 자동 튜닝과 early stopping은 없다.

### 위치별 교차검증과 OOF

`GroupKFold(groups=df.pos)`로 같은 위치의 변이가 서로 다른 fold에 들어가지 않게 한다.
예를 들어 C39Y가 검증에 있으면 C39A를 같은 fold의 학습 데이터로 사용하지 않는다.
이 분할은 같은 단백질의 보지 않은 위치에 대한 평가이며, 다른 단백질로의 일반화나
window가 겹치지 않는 구간별 평가와 동일하지 않다.

각 fold에서 새 모델을 만들고 학습 행으로만 fit한 뒤 검증 행을 예측한다.
모든 행에 대해 자기 fold의 검증 예측을 모은 배열이 OOF(out-of-fold) 예측이다.
`summary.csv`는 이 OOF 전체에서 계산한 지표이며 fold 지표의 단순 평균이 아니다.
모든 feature 조합은 같은 행과 fold를 사용한다.

회귀 지표는 Spearman, RMSE, MAE이고 분류 지표는 AUROC, accuracy다.
분류의 `prediction`은 클래스 1 확률이고 accuracy에는 0.5 기준을 적용한다.
Spearman은 정답/예측이 상수인 경우, AUROC는 검증 정답이 한 클래스인 경우 정의되지 않는다.
이때 해당 지표는 비어 있을 수 있다. 분류 학습 fold에는 두 클래스가 모두 있어야 한다.

교차검증이 끝나면 전체 데이터로 모델을 새로 fit해 `model.joblib`에 저장한다.
이 최종 모델은 추론에 사용하며, OOF 평가는 전체 데이터로 재학습하기 전에 완료된다.
비교 결과를 보고 feature를 고르는 경우 그 선택까지 포함한 성능을 별도로 평가하려면
독립 평가 데이터나 추가 평가 절차가 필요하다. 현재 코드는 별도 test set을 자동 생성하지 않는다.

### 저장 결과

```text
results/
  summary.csv
  config.json
  esm/
    fold_metrics.csv
    oof_predictions.csv
    columns.json
    model.joblib
  esm+charge/
    ...
```

| 파일 | 내용 |
| --- | --- |
| `summary.csv` | 조합별 feature 수와 전체 OOF 성능, compare의 baseline 대비 차이 |
| `config.json` | 실행 옵션, 입력 절대 경로, 정답 열, CV 그룹, 라이브러리 버전 |
| `fold_metrics.csv` | fold별 학습/검증 행 수, 위치 수, 성능 |
| `oof_predictions.csv` | 변이 식별자, 위치, fold, 정답, 검증 예측 |
| `columns.json` | 학습 feature 열의 정확한 순서 |
| `model.joblib` | 전체 데이터로 학습한 모델, 선택 그룹, 열 순서, task, 라이브러리 버전 |

`delta_지표`는 항상 `해당 조합 - 기본 ESM`이다. Spearman/AUROC/accuracy는 양수가 개선,
RMSE/MAE는 음수가 개선이다. 여러 조합의 작업이 모두 성공한 뒤 최종 결과 폴더를 생성한다.
기존 결과 폴더와 예측 CSV를 덮어쓰지 않으므로 다시 실행할 때는 새로운 출력 경로를 사용한다.

저장 모델을 예측에 사용할 때는 저장된 groups로 feature를 재생성하고 columns와 순서를 비교한다.
라이브러리 버전은 기록만 하며 자동으로 실행 환경을 맞춰주지는 않는다.
joblib 모델은 직접 만든 신뢰할 수 있는 파일을 같은 호환 환경에서 불러오는 것을 전제로 한다.

## 명령별 사용 예

아래 파일명은 예시이며 실제 데이터 경로로 바꾼다. 명령은 `feature-lab`에서 실행한다.

### embedding 추출

```powershell
python extract_esm2_embeddings.py
```

### experiment.py
- train 모드 : 한 번에 하나의 feautre로만 학습됨
- compare 모드: 조합마다 모델을 따로 학습하고(없음+각각+전체) 예측 성능을 비교하는 모드
- predict 모드: 이미 학습된 모델로 기능 점수 예측 한 번에 하나의 feature로만 predict

  `--features charge hydropathy structure` 이면
  | 모델 | 사용하는 입력 |
  |---|---|
  | 기준 모델 | ESM |
  | 모델 2 | ESM + 전하 |
  | 모델 3 | ESM + 소수성 |
  | 모델 4 | ESM + 구조(SASA·주변 잔기 수·pLDDT) |
  | 모델 5 | ESM + 전하 + 소수성 + 구조 |
  을 비교함
### 예시
```powershell
# 1. ESM 추출도 할 경우 기본 학습 패키지와 함께 설치
python -m pip install -r requirements-esm.txt

# 2. 전체 mutated_sequence가 있는 원본 표에서 수치 feature 추출
python extract_esm2_embeddings.py --input brca1.csv --output brca1_features.parquet --batch-size 2 --device auto

# mutated_sequence가 없으면 위 명령에 --reference-fasta brca1_wt.fasta를 추가

# 3. ESM + 전하 조합 하나를 5-fold 회귀로 평가하고 최종 모델 저장
python experiment.py train --input brca1_features.parquet --features esm charge --folds 5 --out results_train

# 4. 기본 ESM과 전하/소수성/수소결합 후보를 비교
python experiment.py compare --input brca1_features.parquet --features charge hydropathy hbond --structure AF-P38398-F1-model_v6.cif --out results_compare

# 5. AlphaFold WT CIF로 구조 후보도 포함해 네 후보를 비교
python experiment.py compare --input brca1_features.parquet --features charge hydropathy hbond structure --structure AF-P38398-F1-model_v6.cif --out results_structure

# 6. 새 변이에도 ESM 추출을 먼저 한 뒤 저장된 회귀 모델로 예측
python experiment.py predict --input new_variants_features.parquet --model results_train/esm+charge/model.joblib --out predictions.csv

# 분류 실험은 train/compare에 --task classification을 추가
```

`examples/toy.csv`는 형식과 처리 흐름 확인을 위한 작은 예제다. 실제 BRCA1 성능 검증용이 아니다.
ESM 추출 없이 CLI 흐름만 확인하려면 추가 feature만 선택할 수 있다.

```powershell
python experiment.py train --input examples/toy.csv --features charge --folds 3 --out toy_results
```

예제처럼 행 수가 작으면 LightGBM이 분할을 만들지 않아 상수 예측이나 빈 Spearman이 나올 수 있다.
실제 학습은 Findlay 데이터와 해당 참조 서열에서 추출한 feature를 사용해야 한다.
