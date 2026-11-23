"""실행 흐름 검증용 합성 서열·feature·정답 파일을 만든다.

실제 BRCA1, Findlay 측정값 또는 ESM-2 출력이 아니다. 모델을 다운로드하지 않고
입력 검증과 train/compare/predict의 파일 연결을 확인하기 위한 작은 데이터 세트다.
"""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

# examples/에서 실행해도 프로젝트의 공통 검증/feature 함수를 찾을 수 있게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_esm2_embeddings import reference_sequence
from features import AA, EMBEDDING, design_matrix, variants


def make_variants(reference, positions):
    """각 위치에 네 개의 서로 다른 단일 치환을 만든다. WT와 동일한 치환은 없다."""
    rows = []
    for pos in positions:
        wt = reference[pos - 1]
        for shift in (1, 5, 9, 13):
            mut = AA[(AA.index(wt) + shift) % len(AA)]
            sequence = reference[:pos - 1] + mut + reference[pos:]
            rows.append({"mutant": f"{wt}{pos}{mut}", "mutated_sequence": sequence})
    return pd.DataFrame(rows)


def make_features(raw, wild_vectors, rng):
    """추출기와 같은 열 구조를 만들되 벡터는 난수로 대체한다. 실제 ESM 호출은 없다."""
    df = variants(raw)
    wild = wild_vectors[df.pos.to_numpy() - 1]
    mutant = (wild + 0.35 * rng.normal(size=wild.shape)).astype("float32")
    cosine = np.sum(wild * mutant, axis=1) / (
        np.linalg.norm(wild, axis=1) * np.linalg.norm(mutant, axis=1) + 1e-8)
    metadata = ["mutant", "wt", "pos", "mut", *[key for key in ("DMS_score", "DMS_score_bin") if key in df]]
    result = pd.concat([
        df[metadata].assign(esm_cos_sim=cosine.astype("float32")),
        pd.DataFrame(mutant, columns=EMBEDDING),
    ], axis=1)
    result.attrs["synthetic"] = True
    result.attrs["embedding_source"] = "seeded random vectors; not ESM-2"
    return result


def main():
    """고정 seed로 파일을 생성한다. 기존 examples/synthetic 폴더는 덮어쓰지 않는다."""
    output = Path(__file__).resolve().parent / "synthetic"
    if output.exists():
        raise FileExistsError(f"이미 예제가 있음. 기존 폴더를 덮어쓰지 않음: {output}")

    rng = np.random.default_rng(20261123)
    reference = AA * 4  # 80개 잔기의 인공 서열. BRCA1 서열이 아니다.
    positions = np.arange(1, len(reference) + 1)
    confidence = np.round(np.clip(45 + positions * 0.6 + 5 * np.sin(positions / 5), 0, 100), 2)
    categories = ["H" if score >= 90 else "M" if score >= 70 else "L" if score >= 50 else "D"
                  for score in confidence]

    # 학습은 위치 1~48, 예측 예제는 49~60으로 나눠 변이와 위치가 겹치지 않게 한다.
    train = make_variants(reference, range(1, 49))
    prediction = make_variants(reference, range(49, 61))
    parsed = variants(train)
    chemistry = design_matrix(parsed, ["charge", "hydropathy", "hbond"])
    local_confidence = confidence[parsed.pos.to_numpy() - 1]
    # 실험 측정값 대신 인위적인 규칙에 작은 잡음을 더한다. 생물학적 의미는 없다.
    score = (0.9 - 0.2 * chemistry.delta_hydropathy.abs().to_numpy()
             - 0.45 * chemistry.delta_charge_ph7_lehninger.abs().to_numpy()
             - 0.2 * chemistry.delta_donor.abs().to_numpy()
             - 0.15 * chemistry.delta_acceptor.abs().to_numpy()
             + 0.005 * (local_confidence - 70) + rng.normal(0, 0.08, len(train)))
    train["DMS_score"] = np.round(score, 6)
    train["DMS_score_bin"] = (score >= 0).astype(int)

    # 실제 ESM 없이도 Parquet -> feature 조립 -> LightGBM 경로를 검증할 수 있게 한다.
    wild_vectors = rng.normal(size=(len(reference), len(EMBEDDING))).astype("float32")
    train_features = make_features(train, wild_vectors, rng)
    prediction_features = make_features(prediction, wild_vectors, rng)
    assert reference_sequence(variants(train)) == reference
    assert reference_sequence(variants(prediction)) == reference
    assert set(train.DMS_score_bin) == {0, 1}

    # 모든 파일이 저장된 후 폴더를 공개해 실패한 생성의 중간 산출물이 남지 않게 한다.
    with TemporaryDirectory(prefix="synthetic-staging-", dir=output.parent) as temp:
        staging = Path(temp) / "synthetic"
        staging.mkdir()
        train.to_csv(staging / "variants.csv", index=False)
        prediction.to_csv(staging / "prediction_variants.csv", index=False)
        train_features.to_parquet(staging / "esm_features.parquet", index=False)
        prediction_features.to_parquet(staging / "prediction_features.parquet", index=False)
        (staging / "wt.fasta").write_text(
            ">synthetic_reference_not_BRCA1\n" + reference + "\n", encoding="utf-8")
        (staging / "confidence.json").write_text(json.dumps({
            "synthetic": True,
            "residueNumber": positions.tolist(),
            "confidenceScore": confidence.tolist(),
            "confidenceCategory": categories,
        }, indent=2), encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps({
            "synthetic": True,
            "seed": 20261123,
            "sequence_length": len(reference),
            "train_rows": len(train),
            "prediction_rows": len(prediction),
            "train_positions": [1, 48],
            "prediction_positions": [49, 60],
            "embedding_source": "random vectors with 1280 dimensions; not ESM-2",
            "score_source": "artificial chemistry-based rule plus noise; not Findlay data",
            "confidence_source": "artificial numbers; not AlphaFold predictions",
        }, indent=2), encoding="utf-8")
        staging.rename(output)
    print(f"Created {len(train)} training variants and {len(prediction)} prediction variants: {output}")


if __name__ == "__main__":
    main()
