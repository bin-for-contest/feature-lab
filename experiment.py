"""ESM-2와 추가 feature의 조합을 LightGBM으로 평가하는 실행 진입점.

train은 지정한 조합 하나를, compare는 기본 ESM과 후보를 추가한 조합들을 학습한다.
두 명령 모두 위치별 교차검증 예측으로 성능을 계산한 뒤 전체 데이터로 최종 모델을
다시 학습한다. predict는 저장한 모델을 읽어 정답이 없는 새 변이에도 점수를 붙인다.
서열 임베딩 자체는 여기서 만들지 않고 extract_esm2_embeddings.py의 출력을 읽는다.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from features import CANDIDATES, GROUPS, add_structure, design_matrix, read_table, variants


def write_json(path, obj):
    """설정/열 목록을 UTF-8 JSON으로 저장하고 NaN/Infinity 직렬화는 거부한다."""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y, pred, task):
    """정답 y와 예측 pred로 task에 맞는 지표 사전을 반환한다.

    회귀: Spearman은 순위 일치도(클수록 좋음), RMSE/MAE는 점수 오차(작을수록 좋음).
    분류: AUROC는 확률 순위, accuracy는 0.5를 기준으로 나눈 클래스의 정확도다.
    수학적으로 정의되지 않는 일부 지표는 NaN 대신 None을 사용한다.
    """
    if task == "classification":
        # score는 P(DMS_score_bin=1). 한 클래스만 있는 평가 fold의 AUROC는 정의되지 않음.
        return {"auroc": float(roc_auc_score(y, pred)) if np.unique(y).size == 2 else None,
                "accuracy": float(np.mean((pred >= 0.5) == y))}
    # 정답 또는 예측이 상수이면 순위 상관이 정의되지 않는다.
    rho = None
    if np.unique(y).size > 1 and np.unique(pred).size > 1:
        rho = float(spearmanr(y, pred).statistic)
    return {"spearman": rho, "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred))}


def predict(estimator, X, task):
    """회귀는 연속 기능 점수, 분류는 라벨 1의 확률을 1차원 배열로 반환한다."""
    if task == "regression":
        return estimator.predict(X)
    # 학습 라벨은 run()에서 {0, 1}로 제한하므로 두 번째 확률 열이 클래스 1에 해당한다.
    return estimator.predict_proba(X)[:, 1]


def evaluate(df, groups, y, folds, args, output):
    """feature 조합 하나를 교차검증하고 산출물을 저장한다.

    df와 y는 같은 행 순서여야 하며 folds는 (학습 행 인덱스, 검증 행 인덱스) 목록이다.
    매 fold에서 검증 위치의 라벨을 보지 않은 모델로 예측해 OOF 벡터를 완성한다.
    반환값은 모든 OOF 예측을 합쳐 계산한 조합별 요약이며 fold 점수의 단순 평균이 아니다.
    마지막에 전체 데이터로 학습한 모델은 추론용이고, 그 모델의 학습 성능은 보고하지 않는다.
    """
    X = design_matrix(df, groups)
    # 같은 모델 설정을 모든 feature 조합에 적용해 feature 추가 효과를 비교한다.
    # random_state는 0으로 고정하고 나머지는 LightGBM 기본 설정을 사용한다.
    # 현재 하이퍼파라미터 탐색이나 early stopping은 수행하지 않는다.
    head = (lgb.LGBMRegressor(random_state=0, verbosity=-1) if args.task == "regression"
            else lgb.LGBMClassifier(random_state=0, verbosity=-1))
    # 단계가 하나인 Pipeline을 저장해 fit/predict와 sklearn clone 인터페이스를 통일한다.
    pipeline = make_pipeline(head)
    # OOF(out-of-fold): 각 행이 자신의 검증 fold에서 얻은 예측을 저장하는 배열.
    # GroupKFold가 모든 행을 정확히 한 번 검증에 사용하므로 마지막에는 빈 칸이 없다.
    oof = np.empty(len(df))
    fold_ids = np.zeros(len(df), dtype=int)
    scores = []
    for fold, (train, valid) in enumerate(folds, start=1):
        if args.task == "classification" and np.unique(y[train]).size != 2:
            raise ValueError(f"fold {fold}의 학습 위치에 두 클래스가 모두 필요함")
        # 각 fold마다 새 모델을 fit함. LightGBM은 feature 표준화 없이 사용함.
        # clone은 설정만 복사하고 학습 상태는 가져오지 않아 이전 fold의 모델을 재사용하지 않는다.
        fitted = clone(pipeline).fit(X.iloc[train], y[train])
        oof[valid] = predict(fitted, X.iloc[valid], args.task)
        fold_ids[valid] = fold
        # 행 수와 고유 위치 수를 함께 남겨 각 fold가 어떤 크기로 평가됐는지 추적한다.
        scores.append({"fold": fold, "n_train": len(train), "n_valid": len(valid),
                       "train_positions": df.pos.iloc[train].nunique(),
                       "valid_positions": df.pos.iloc[valid].nunique(),
                       **metrics(y[valid], oof[valid], args.task)})

    # 조합별 폴더에 평가 근거와 추론용 모델을 함께 저장한다.
    output.mkdir()
    columns = X.columns.tolist()
    pd.DataFrame(scores).to_csv(output / "fold_metrics.csv", index=False)
    # 원래 변이 식별자와 fold를 보존해 오차 분석이나 조합 간 행별 비교에 사용할 수 있다.
    df[["mutant", "pos"]].assign(fold=fold_ids, observed=y, prediction=oof).to_csv(
        output / "oof_predictions.csv", index=False)
    write_json(output / "columns.json", columns)
    # OOF 평가를 끝낸 후 전체 행으로 다시 fit한 모델은 새로운 변이의 추론용으로 보관함.
    final_model = pipeline.fit(X, y)
    # groups는 feature 재생성 규칙, columns는 실제 열 순서다. 둘 다 추론 시 필요하다.
    # 라이브러리 버전은 저장 환경을 추적하는 기록이며 자동 호환성 검사는 아니다.
    joblib.dump({"model": final_model, "groups": groups, "columns": columns,
                 "task": args.task, "sklearn_version": sklearn.__version__,
                 "lightgbm_version": lgb.__version__}, output / "model.joblib")
    return {"features": "+".join(groups), "n_rows": len(df), "n_features": len(columns),
            **metrics(y, oof, args.task)}


def run(args):
    """train/compare의 입력 검증, 공통 fold 생성, 조합 평가, 결과 확정을 수행한다."""
    if args.out.exists():
        raise ValueError(f"출력 폴더가 이미 있음. 새 --out을 지정할 것: {args.out}")
    # 파일의 행 순서가 바뀌어도 같은 정렬 기준에서 fold와 결과를 만들도록 한다.
    # 이는 학습/평가용 정렬이며 predict 명령은 사용자가 준 입력 순서를 유지한다.
    df = variants(read_table(args.input)).sort_values("mutant").reset_index(drop=True)
    if args.structure:
        df = add_structure(df, args.structure, include_hbond="hbond" in args.features)
    # 회귀의 정답은 Findlay 기능 점수, 분류의 정답은 입력 파일에 제공된 이진 라벨이다.
    # 분류 라벨을 여기서 임의의 임계값으로 새로 만들지는 않는다.
    label = "DMS_score" if args.task == "regression" else "DMS_score_bin"
    if label not in df:
        raise ValueError(f"학습 입력에 {label} 열이 필요함")
    y = pd.to_numeric(df[label], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"{label} 열에 결측/무한값이 있음")
    if args.task == "classification" and set(np.unique(y)) != {0, 1}:
        raise ValueError("이진 분류에는 0과 1 라벨이 모두 필요함")
    if not 2 <= args.folds <= df.pos.nunique():
        raise ValueError("--folds는 2 이상이며 고유 pos 수 이하여야 함")
    structure_groups = [name for name in args.features if name in {"hbond", "structure"}]
    if structure_groups:
        # JSON만 준 경우처럼 구조 열이 부족하면 baseline 학습을 시작하기 전에 알려준다.
        design_matrix(df, structure_groups)
    # C39Y와 C39A처럼 같은 위치의 치환을 학습/검증으로 나눠 넣지 않는다.
    # 따라서 평가는 이 단백질의 '보지 않은 위치'에 대한 예측을 측정한다.
    # 같은 실행의 모든 조합이 같은 행·같은 fold 인덱스를 재사용한다.
    # 인접 위치의 서열 window 중첩이나 다른 단백질로의 일반화까지 검증하는 분할은 아니다.
    folds = list(GroupKFold(n_splits=args.folds, shuffle=True, random_state=42).split(df, y, groups=df.pos))
    if args.command == "compare":
        # 예: charge, hydropathy를 고르면 esm / esm+charge / esm+hydropathy /
        # esm+charge+hydropathy를 실행한다. 모든 부분집합을 탐색하는 코드는 아니다.
        base = ["esm"]
        setups = [base, *[base + [name] for name in args.features]]
        if len(args.features) > 1:
            setups.append(base + args.features)
    else:
        # train은 지정한 feature만 사용하므로 esm 없는 후보 단독 실험도 가능하다.
        setups = [args.features]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 모든 조합이 성공한 뒤 결과 폴더를 공개함. 실패한 실행을 정상 결과와 구분함.
    with TemporaryDirectory(dir=args.out.parent) as temp:
        # 최종 폴더와 같은 부모 아래에서 작업해 완성된 폴더를 rename으로 옮길 수 있게 한다.
        # 중간 조합이 실패하면 임시 결과는 정리되고 최종 --out 폴더는 생성되지 않는다.
        staging = Path(temp) / "results"
        staging.mkdir()
        summaries = []
        for groups in setups:
            print(f"Evaluating {'+'.join(groups)}", flush=True)
            summaries.append(evaluate(df, groups, y, folds, args, staging / "+".join(groups)))
        summary = pd.DataFrame(summaries)
        if args.command == "compare":
            keys = ["spearman", "rmse", "mae"] if args.task == "regression" else ["auroc", "accuracy"]
            for key in keys:
                numeric = pd.to_numeric(summary[key], errors="coerce")
                # delta는 항상 '해당 조합 - 기본 ESM'이다.
                # Spearman/AUROC/accuracy는 양수가 개선, RMSE/MAE는 음수가 개선이다.
                summary[f"delta_{key}"] = numeric - numeric.iloc[0]
        summary.to_csv(staging / "summary.csv", index=False)
        # 실행 옵션과 절대 경로를 남겨 어떤 파일/설정으로 결과를 만들었는지 확인한다.
        config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update({"target": label, "groups_for_cv": "pos", "sklearn_version": sklearn.__version__,
                       "lightgbm_version": lgb.__version__})
        write_json(staging / "config.json", config)
        staging.rename(args.out)
    print(summary.to_string(index=False))


def infer(args):
    """저장 모델의 feature 규칙을 적용해 입력 순서대로 예측 CSV를 만든다.

    정답 열은 필요 없지만, 학습에서 선택한 ESM 열이나 구조 feature는 제공해야 한다.
    feature 정의가 달라져 열 이름/순서가 바뀌면 잘못된 모델 입력을 허용하지 않는다.
    """
    # joblib은 동일 환경에서 직접 생성한 모델 파일을 로드하는 용도임.
    saved = joblib.load(args.model)
    legacy_hbond = {"delta_donor", "delta_acceptor"} <= set(saved["columns"])
    df = variants(read_table(args.input))
    if args.structure:
        df = add_structure(df, args.structure, include_hbond="hbond" in saved["groups"] and not legacy_hbond)
    # 추론 CLI에서 feature를 다시 고르지 않고 학습 시 저장한 조합을 그대로 사용한다.
    X = design_matrix(df, saved["groups"], legacy_hbond=legacy_hbond)
    if X.columns.tolist() != saved["columns"]:
        raise ValueError("현재 feature 순서가 학습 시점과 다름")
    pred = predict(saved["model"], X, saved["task"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # mode='x'는 기존 CSV를 덮어쓰지 않고, 경로가 이미 있으면 실패하게 한다.
    df[["mutant", "pos"]].assign(prediction=pred).to_csv(args.out, index=False, mode="x")
    print(f"{len(df)} predictions -> {args.out}")


def main():
    """명령줄 옵션을 정의하고 train/compare/predict 실행 함수로 분기한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "compare", "predict"):
        # 세 명령이 공통으로 받는 입력/출력/구조 옵션을 먼저 정의한다.
        cmd = commands.add_parser(name)
        cmd.add_argument("--input", required=True, type=Path, help="변이 단위 CSV/TSV/Parquet")
        cmd.add_argument("--out", required=True, type=Path, help="새 결과 폴더(predict는 CSV 파일)")
        cmd.add_argument("--structure", type=Path,
                         help="hbond/structure 그룹용 AlphaFold WT CIF 또는 위치별 구조 표. JSON/PDB는 pLDDT 단독용")
        if name == "predict":
            # 예측 작업의 task와 feature는 모델 파일 안에 있으므로 --model만 추가한다.
            cmd.add_argument("--model", required=True, type=Path)
            continue
        # train은 자유 조합, compare는 고정 ESM baseline에 더할 후보만 받는다.
        candidates = GROUPS if name == "train" else CANDIDATES
        cmd.add_argument("--features", nargs="+", choices=candidates, required=name == "compare",
                         default=["esm"], help="공백으로 구분한 feature 그룹")
        cmd.add_argument("--task", choices=("regression", "classification"), default="regression")
        cmd.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    try:
        if args.command == "predict":
            infer(args)
        elif len(args.features) != len(set(args.features)):
            parser.error("--features에 중복된 그룹이 있음")
        else:
            run(args)
    # 입력/경로 등 예상 가능한 오류는 짧은 메시지와 종료 코드 2로 보고한다.
    # 그 밖의 예외는 traceback을 남겨 구현 오류나 환경 문제를 조사할 수 있게 한다.
    except (ValueError, FileExistsError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"오류: {exc}\n")


if __name__ == "__main__":
    main()
