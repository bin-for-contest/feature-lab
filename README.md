# BRCA1 feature 실험

ESM-2 feature에 추가 특성을 결합하고 LightGBM으로 Findlay DMS 기능 점수를 예측한다.
동일 위치의 변이는 같은 fold에 배정하며, 모든 비교 조합은 같은 fold를 사용한다.

## 추가 feature 후보

| CLI 이름 | 값과 출처 |
| --- | --- |
| `charge` | PyPI `peptides.Peptide.charge(pH=7.0, pKscale="Lehninger")`의 Mut − WT 차이 |
| `hydropathy` | `peptides.Peptide.hydrophobicity(scale="KyteDoolittle")`의 Mut − WT 차이 |
| `plddt` | 별도 AlphaFold WT PDB 또는 `pos,plddt` 표에서 가져온 위치별 예측 신뢰도 |
| `hbond` | 기존 잔기별 donor/acceptor 소속 여부의 Mut − WT 차이. 실제 수소결합 수가 아닌 근사치 |

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
`--features`에 `plddt`를 추가하고 `--structure`로 AlphaFold WT PDB 또는 위치별 표를 지정한다.
기본 `esm`에는 임베딩 1,280개와 코사인 유사도, 총 1,281개 feature가 포함된다.
`esm_llr`은 추출·학습에서 제외한다. 기존 입력 파일에 이 열이 남아 있어도 사용하지 않는다.
비교는 기본 ESM, 각 후보를 하나씩 추가한 조합, 선택한 후보를 모두 추가한 조합을 실행한다.
LLR을 포함해 학습한 기존 ESM 모델은 새 feature 구성으로 다시 학습해야 한다.

전하 계산을 기존 정수 근사에서 pH 기반 계산으로 바꿨으므로 `charge`를 사용한 기존 모델은
다시 학습해야 한다. 새 열 이름은 `delta_charge_ph7_lehninger`이며 기존 열 이름과 다르면
예측 단계에서 오류로 중단한다.
