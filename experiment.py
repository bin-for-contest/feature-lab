"""BRCA1 feature ablation을 실행하고 위치별 교차검증과 최종 모델을 저장함."""

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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y, pred, task):
    if task == "classification":
        # score는 P(DMS_score_bin=1). 한 클래스만 있는 평가 fold의 AUROC는 정의되지 않음.
        return {"auroc": float(roc_auc_score(y, pred)) if np.unique(y).size == 2 else None,
                "accuracy": float(np.mean((pred >= 0.5) == y))}
    rho = None
    if np.unique(y).size > 1 and np.unique(pred).size > 1:
        rho = float(spearmanr(y, pred).statistic)
    return {"spearman": rho, "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred))}


def predict(estimator, X, task):
    if task == "regression":
        return estimator.predict(X)
    return estimator.predict_proba(X)[:, 1]


def evaluate(df, groups, y, folds, args, output):
    """각 행을 해당 위치를 한 번도 보지 않은 모델로 예측해 OOF 벡터를 완성함."""
    X = design_matrix(df, groups)
    head = (lgb.LGBMRegressor(random_state=0, verbosity=-1) if args.task == "regression"
            else lgb.LGBMClassifier(random_state=0, verbosity=-1))
    pipeline = make_pipeline(head)
    oof = np.empty(len(df))
    fold_ids = np.zeros(len(df), dtype=int)
    scores = []
    for fold, (train, valid) in enumerate(folds, start=1):
        if args.task == "classification" and np.unique(y[train]).size != 2:
            raise ValueError(f"fold {fold}의 학습 위치에 두 클래스가 모두 필요함")
        # 각 fold마다 새 모델을 fit함. LightGBM은 feature 표준화 없이 사용함.
        fitted = clone(pipeline).fit(X.iloc[train], y[train])
        oof[valid] = predict(fitted, X.iloc[valid], args.task)
        fold_ids[valid] = fold
        scores.append({"fold": fold, "n_train": len(train), "n_valid": len(valid),
                       "train_positions": df.pos.iloc[train].nunique(),
                       "valid_positions": df.pos.iloc[valid].nunique(),
                       **metrics(y[valid], oof[valid], args.task)})

    output.mkdir()
    columns = X.columns.tolist()
    pd.DataFrame(scores).to_csv(output / "fold_metrics.csv", index=False)
    df[["mutant", "pos"]].assign(fold=fold_ids, observed=y, prediction=oof).to_csv(
        output / "oof_predictions.csv", index=False)
    write_json(output / "columns.json", columns)
    # OOF 평가를 끝낸 후 전체 행으로 다시 fit한 모델은 새로운 변이의 추론용으로 보관함.
    final_model = pipeline.fit(X, y)
    joblib.dump({"model": final_model, "groups": groups, "columns": columns,
                 "task": args.task, "sklearn_version": sklearn.__version__,
                 "lightgbm_version": lgb.__version__}, output / "model.joblib")
    return {"features": "+".join(groups), "n_rows": len(df), "n_features": len(columns),
            **metrics(y, oof, args.task)}


def run(args):
    if args.out.exists():
        raise ValueError(f"출력 폴더가 이미 있음. 새 --out을 지정할 것: {args.out}")
    df = variants(read_table(args.input)).sort_values("mutant").reset_index(drop=True)
    if args.structure:
        df = add_structure(df, args.structure)
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
    # 같은 실행의 모든 조합이 같은 행·같은 fold 인덱스를 재사용함.
    folds = list(GroupKFold(n_splits=args.folds).split(df, y, groups=df.pos))
    if args.command == "compare":
        base = ["esm"]
        setups = [base, *[base + [name] for name in args.features]]
        if len(args.features) > 1:
            setups.append(base + args.features)
    else:
        setups = [args.features]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 모든 조합이 성공한 뒤 결과 폴더를 공개함. 실패한 실행을 정상 결과와 구분함.
    with TemporaryDirectory(dir=args.out.parent) as temp:
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
                summary[f"delta_{key}"] = numeric - numeric.iloc[0]
        summary.to_csv(staging / "summary.csv", index=False)
        config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update({"target": label, "groups_for_cv": "pos", "sklearn_version": sklearn.__version__,
                       "lightgbm_version": lgb.__version__})
        write_json(staging / "config.json", config)
        staging.rename(args.out)
    print(summary.to_string(index=False))


def infer(args):
    # joblib은 동일 환경에서 직접 생성한 모델 파일을 로드하는 용도임.
    saved = joblib.load(args.model)
    df = variants(read_table(args.input))
    if args.structure:
        df = add_structure(df, args.structure)
    X = design_matrix(df, saved["groups"])
    if X.columns.tolist() != saved["columns"]:
        raise ValueError("현재 feature 순서가 학습 시점과 다름")
    pred = predict(saved["model"], X, saved["task"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["mutant", "pos"]].assign(prediction=pred).to_csv(args.out, index=False, mode="x")
    print(f"{len(df)} predictions -> {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "compare", "predict"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--input", required=True, type=Path, help="변이 단위 CSV/TSV/Parquet")
        cmd.add_argument("--out", required=True, type=Path, help="새 결과 폴더(predict는 CSV 파일)")
        cmd.add_argument("--structure", type=Path, help="AlphaFold WT PDB 또는 pos,plddt 표")
        if name == "predict":
            cmd.add_argument("--model", required=True, type=Path)
            continue
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
    except (ValueError, FileExistsError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"오류: {exc}\n")


if __name__ == "__main__":
    main()
